"""
LangChain RAG Agent Harness: Tools, Middleware, Guardrails, and Memory Management.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from rag_engine import RAGEngine, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a distinguished, encouraging, and expert Professor teaching university students.
Your mission is to mentor students, explain complex subjects with supreme clarity, and teach how to solve problems step-by-step using clear intuitive explanations and visual representations.

Your ONLY source of curriculum and factual knowledge is the uploaded course PDF documents accessed via the `search_pdf_documents` tool.

MANDATORY RESPONSE FORMAT & TEACHING PROTOCOL:
For ANY question, concept, or problem covered in the uploaded course materials, you MUST ALWAYS provide:
1. A rigorous STEP-BY-STEP ANALYSIS broken down into explicit numbered sections.
2. A clean, visual MERMAID DIAGRAM illustrating the process, relationships, formula dependencies, or workflow.

STRUCTURE OF EVERY RESPONSE:
1. **Core Concept & Big Picture Intuition**:
   - Give the student an intuitive, high-level understanding of what this problem or concept is about before technical notation.

2. **Step-by-Step Problem Analysis**:
   - **Step 1: Problem Definition & Givens**: Identify what is given in the course material, parameters, and the exact objective.
   - **Step 2: Governing Principles & Formulas**: State the definitions, theorems, or mathematical laws using formatted LaTeX (inline `$formula$` or block `$$equation$$`).
   - **Step 3: Step-by-Step Derivation & Analytical Solution**: Work through each calculation, reasoning step, or proof sequentially with full justification for each step.
   - **Step 4: Summary Table or Key Parameters**: Summarize values, boundary conditions, or variables in a clean markdown table.

3. **Visual Mermaid Diagram (MANDATORY)**:
   - You MUST include at least one valid ```mermaid ... ``` code block in every response.
   - Always start with `graph TD` on the first line.
   - Use simple single-word alphanumeric node IDs: A, B, C, D or Step1, Step2, Step3.
   - CRITICAL SYNTAX RULE: ALWAYS put node text in double quotes: `A["Step 1: Hypothesis"] --> B["Step 2: Conclusion"]`.
   - NEVER use raw parentheses `()`, brackets `[]`, braces `{}`, or math symbols `<`, `>`, `&`, `|` outside of double quotes.
   - Keep node labels concise (under 10 words per node) and cleanly connected with `-->`.

4. **Summary Takeaway**:
   - Provide a concise summary of the key lesson and exam takeaways the student must remember.

STRICT GROUNDING GUARDRAIL:
- You can ONLY teach facts, data, numbers, and solutions substantiated by the uploaded PDF documents.
- If the requested topic, question, or problem is NOT covered in the uploaded documents (or search yields no matches), warmly inform the student:
  "I don't have information about that in the uploaded documents."
- Never speculate, hallucinate facts, or invent outside curriculum not present in the files.
- For every theorem, principle, equation, and solution step, cite the exact source file and page: [Source: <filename>, Page: <page>].
"""


class RAGAgentService:
    """Encapsulates the LangChain agent harness with memory, error handling, and guardrails."""

    def __init__(
        self,
        rag_engine: RAGEngine,
        model_name: str = "qwen3.5:9b",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.rag_engine = rag_engine
        self.model_name = model_name
        self.ollama_url = ollama_url

        # Checkpointer for conversational memory across turns
        self.checkpointer = MemorySaver()

        # Mutable container for the latest retrieved citations (shared with tool closure)
        self._latest_citations: List[RetrievedChunk] = []

        # Build tools and middleware
        self._tools = self._build_tools()
        self._middleware = self._build_middleware()

        # Initialize LLM based on available provider credentials
        use_gemini_only = os.getenv("USE_GEMINI_ONLY", "").lower() in ("true", "1", "yes") or bool(os.getenv("RAILWAY_ENVIRONMENT"))
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if use_gemini_only or gemini_api_key:
            self.provider = "gemini"
            chat_model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
            self.model_name = chat_model_name
            if gemini_api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI

                logger.info("Using Google Generative AI LLM (%s)", chat_model_name)
                self.llm = ChatGoogleGenerativeAI(
                    model=chat_model_name,
                    google_api_key=gemini_api_key,
                    temperature=0.0,
                )
            else:
                logger.warning("Gemini provider selected on Railway/Cloud, but GEMINI_API_KEY is not yet set.")
                from langchain_core.language_models.fake import FakeListLLM

                self.llm = FakeListLLM(
                    responses=["GEMINI_API_KEY is required on Railway. Please enter your key in Railway Variables or the UI."]
                )
        else:
            logger.info("Using Ollama Chat (%s) at %s", self.model_name, self.ollama_url)
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_url,
                temperature=0.0,  # Zero temperature for deterministic, factual output
            )
            self.provider = "ollama"

        # Initialize Agent Harness
        self.agent = create_agent(
            model=self.llm,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            middleware=self._middleware,
        )

    def _build_tools(self) -> list:
        rag = self.rag_engine
        # Use mutable list reference so the tool closure can write citations
        # and ask() can read them — no thread_id plumbing needed.
        citations_ref = self._latest_citations

        @tool
        def search_pdf_documents(query: str) -> str:
            """Search the uploaded PDF documents for relevant excerpts matching the query.

            Args:
                query: The natural language search query or keywords.

            Returns:
                Formatted relevant document chunks with source filenames and page numbers,
                or a message stating that no relevant information was found.
            """
            try:
                chunks = rag.retrieve(query=query, top_k=4)
                if not chunks:
                    return (
                        "No relevant information found in uploaded documents. "
                        "The query does not match any content in the uploaded PDFs."
                    )

                # Store citations via mutable reference for the UI to pick up
                citations_ref.clear()
                citations_ref.extend(chunks)

                formatted_chunks = []
                for idx, chunk in enumerate(chunks, start=1):
                    formatted_chunks.append(
                        f"--- Excerpt {idx} [Source: {chunk.source}, Page: {chunk.page}, Confidence: {chunk.score}] ---\n"
                        f"{chunk.content}\n"
                    )

                return "\n".join(formatted_chunks)

            except Exception as e:
                logger.error("Error executing document search tool: %s", e)
                return f"Error executing document search: {e}"

        return [search_pdf_documents]

    def _build_middleware(self) -> list:
        """Construct middleware for error handling, timeouts, and logging."""

        @wrap_tool_call
        def error_handling_middleware(request, handler):
            """Middleware intercepting tool calls to enforce resilience and graceful failure."""
            tool_name = getattr(request, "name", "unknown_tool")
            logger.info("Executing tool: %s with args: %s", tool_name, getattr(request, "args", {}))
            try:
                response = handler(request)
                return response
            except Exception as exc:
                logger.error("Tool '%s' encountered an unhandled exception: %s", tool_name, exc)
                # Return graceful error string instead of halting the entire agent loop
                from langchain_core.messages import ToolMessage

                tool_call_id = getattr(request, "tool_call_id", "fallback_id")
                return ToolMessage(
                    content=f"Tool error: Unable to complete search due to: {exc}. Please inform the user gracefully.",
                    tool_call_id=tool_call_id,
                )

        return [error_handling_middleware]

    def ask(self, user_question: str, thread_id: str = "default_session") -> Dict[str, Any]:
        """
        Invoke the agent harness with conversational memory.
        Returns the final answer and any retrieved citations.
        """
        # Clear citations for this turn
        self._latest_citations.clear()

        # Early guard: if no documents are indexed, don't waste an LLM call
        if self.rag_engine.document_count == 0:
            return {
                "answer": "No documents have been uploaded yet. Please upload a PDF first, then ask your question.",
                "citations": [],
                "thread_id": thread_id,
            }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=user_question)]},
                config=config,
            )

            messages: List[BaseMessage] = result.get("messages", [])
            final_message = messages[-1] if messages else None

            # Extract content text
            answer_text = ""
            if final_message:
                if isinstance(final_message.content, str):
                    answer_text = final_message.content
                elif isinstance(final_message.content, list):
                    # Handle content blocks
                    text_parts = [
                        block.get("text", "")
                        for block in final_message.content
                        if isinstance(block, dict) and "text" in block
                    ]
                    answer_text = "\n".join(text_parts) if text_parts else str(final_message.content)
            else:
                answer_text = "I don't have information about that in the uploaded documents."

            # Snapshot citations captured by the tool during this turn
            citations = list(self._latest_citations)

            return {
                "answer": answer_text.strip(),
                "citations": citations,
                "thread_id": thread_id,
            }

        except Exception as e:
            logger.error("Agent execution error on thread %s: %s", thread_id, e)
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
                return {
                    "answer": (
                        "⚠️ **Invalid Google Gemini API Key**\n\n"
                        "Google rejected the API key (`400 API_KEY_INVALID`).\n\n"
                        "💡 **How to fix:**\n"
                        "1. Generate a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).\n"
                        "2. Note: Valid Gemini API keys always start with `AIzaSy...` (39 characters).\n"
                        "3. Update `GEMINI_API_KEY` in your Railway Variables or in the sidebar Settings tab."
                    ),
                    "citations": [],
                    "thread_id": thread_id,
                    "error": err_str,
                }
            if "Connection refused" in err_str or "ConnectError" in err_str or "11434" in err_str:
                return {
                    "answer": (
                        "⚠️ **Model Provider Connection Error**\n\n"
                        f"Unable to reach the local Ollama instance at `{self.ollama_url}`.\n\n"
                        "💡 **Fix:** If running in the cloud (Railway) or your local Ollama is offline, "
                        "please enter a **Google Gemini API Key** in the sidebar Settings tab or prompt above to switch to cloud inference instantly."
                    ),
                    "citations": [],
                    "thread_id": thread_id,
                    "error": err_str,
                }
            return {
                "answer": f"An error occurred while processing your question: {e}",
                "citations": [],
                "thread_id": thread_id,
                "error": str(e),
            }
