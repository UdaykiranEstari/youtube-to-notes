"""AI-powered transcript analysis and structured note generation.

Uses an :class:`~src.modules.llm_providers.LLMProvider` to send video
transcripts to an LLM and parse the structured JSON response into notes
with inline screenshot timestamps.
"""

import os
import json
from typing import Dict, List, Optional

from src.modules.llm_providers import LLMProvider, create_provider


class ContentAnalyzer:
    """Analyzes video transcripts and produces structured notes via an LLM.

    The analyzer sends crafted prompts to the configured
    :class:`~src.modules.llm_providers.LLMProvider` and parses the JSON
    response into a dict with ``title``, ``summary``, and ``sections``
    (each containing interleaved text and screenshot entries).

    Args:
        provider: LLM backend to use for generation.  Falls back to
            Vertex AI when *None*.
    """

    def __init__(self, provider: LLMProvider = None):
        """Initialize ContentAnalyzer with an LLM provider.

        Args:
            provider: LLMProvider instance. If *None*, defaults to Vertex AI.
        """
        if provider is None:
            # Default to Vertex AI for backwards compatibility
            self.provider = create_provider("vertex_ai")
        else:
            self.provider = provider

    def analyze_transcript(self, transcript: str, video_title: str = "", word_timestamps: Optional[List[Dict]] = None, screenshot_density: str = "Medium", detail_level: str = "Detailed", raw_output_path: Optional[str] = None) -> Dict:
        """
        Analyzes the transcript and returns structured notes with inline screenshot placement.
        
        Args:
            transcript: The video transcript text
            video_title: Title of the video
            word_timestamps: Optional word-level timestamps from audio transcription
            screenshot_density: "Low", "Medium", or "High"
            detail_level: "Brief", "Standard", or "Detailed"
            raw_output_path: Optional path to save the raw Gemini response text
        
        Returns:
            Dict with inline content structure
        """
        
        # Check if raw response exists to resume/retry
        if raw_output_path and os.path.exists(raw_output_path):
            print(f"Found existing raw response at {raw_output_path}. Attempting to process...")
            try:
                with open(raw_output_path, "r", encoding="utf-8") as f:
                    text_response = f.read()
                
                # Try to parse/repair
                return self._process_response_text(text_response)
            except Exception as e:
                print(f"Failed to process existing raw response: {e}. Regenerating...")

        if word_timestamps:
            # Use word-level timestamps for precise inline placement
            prompt = self._create_inline_prompt(transcript, video_title, word_timestamps, screenshot_density, detail_level)
        else:
            # Fallback to section-based timestamps
            prompt = self._create_section_prompt(transcript, video_title, screenshot_density, detail_level)
        
        response_dict = self.provider.generate_json(prompt, max_tokens=16384)
        
        # Save raw response if path provided
        if raw_output_path:
            try:
                with open(raw_output_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(response_dict, indent=2))
                print(f"Raw LLM response saved to: {raw_output_path}")
            except Exception as e:
                print(f"Failed to save raw response: {e}")
        
        return response_dict if response_dict else {}

    def _process_response_text(self, text_response: str) -> Dict:
        """Parse raw LLM text into a dict, repairing truncated JSON if needed.

        Args:
            text_response: Raw text returned by the LLM (may include markdown
                code fences).

        Returns:
            Parsed JSON dict, or empty dict on failure.
        """
        # Clean up markdown code blocks if present
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        elif text_response.startswith("```"):
            text_response = text_response[3:]
            
        if text_response.endswith("```"):
            text_response = text_response[:-3]
            
        text_response = text_response.strip()
        
        try:
            return json.loads(text_response)
        except json.JSONDecodeError:
            print("Error decoding JSON. Attempting to repair truncated response...")
            repaired_json = self._repair_json(text_response)
            try:
                return json.loads(repaired_json)
            except json.JSONDecodeError:
                print("Failed to repair JSON. Raw response saved to file.")
                return {}

    def _repair_json(self, json_str: str) -> str:
        """
        Attempts to repair a truncated JSON string by closing open strings, objects, and arrays.
        """
        # Fix common syntax errors
        json_str = json_str.replace(':- "', ': "')
        
        stack = []
        escaped = False
        in_string = False
        
        for char in json_str:
            if char == '\\':
                escaped = not escaped
                continue
            
            if char == '"' and not escaped:
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char == '}' or char == ']':
                    if stack:
                        # If the character matches the expected closer, pop it
                        if stack[-1] == char:
                            stack.pop()
                        # If it doesn't match, we might have a malformed JSON or logic error,
                        # but for truncation repair, we just ignore mismatches in the middle
                        # as we assume the start is valid.
            
            escaped = False
            
        # Close the string if open
        if in_string:
             json_str += '"'
        
        # Append remaining closers in reverse order
        while stack:
            json_str += stack.pop()
            
        return json_str
    
    def _get_density_instruction(self, density: str) -> str:
        """Return a prompt fragment describing the target screenshot density.

        Args:
            density: One of ``"Low"``, ``"Medium"``, or ``"High"``.

        Returns:
            Instruction string to embed in the LLM prompt.
        """
        if density == "Low":
            return "Aim for **4-6 screenshots** per major section. Use an **even number** of screenshots per section (2, 4, 6, …) so they pair into a 2-column grid."
        elif density == "High":
            return "Aim for **14-20 screenshots** per major section. Capture every visual step. Use an **even number** of screenshots per section (2, 4, 6, …) so they pair into a 2-column grid."
        else: # Medium
            return "Aim for **8-12 screenshots** per major section. Use an **even number** of screenshots per section (2, 4, 6, …) so they pair into a 2-column grid."

    def _get_detail_instruction(self, level: str) -> str:
        """Return a prompt fragment describing the target detail level.

        Args:
            level: One of ``"Brief"``, ``"Standard"``, ``"Detailed"``, or
                ``"More Detailed"``.

        Returns:
            Instruction string to embed in the LLM prompt.
        """
        if level == "Brief":
            return """Create **concise summaries** and bullet points. Focus only on key takeaways. Be brief.
            
**Writing Style**: Use clear, objective language. State information directly without referencing the speaker or author."""
        elif level == "Standard":
            return """Create **balanced notes**. Explain concepts clearly but avoid excessive detail.
            
**Writing Style**: Write in objective, textbook style. State facts directly (e.g., "The process involves..." not "The author explains that the process involves..."). Do not reference the speaker, author, or video."""
        elif level == "Detailed":
            return """Create **comprehensive, detailed notes** that capture all technical concepts, steps, examples, and nuances. Explain things thoroughly.
            
**Writing Style**: Use professional, educational language. State information objectively without mentioning the speaker or author. Write as if creating textbook content."""
        else: # More Detailed
            return """Create **exhaustive, textbook-quality notes** with maximum depth:
- Explain concepts in detail with background context
- Include all technical steps, examples, and edge cases
- Provide thorough explanations as if teaching the topic from scratch
- Break down complex ideas into clear, digestible sub-sections
- Add relevant context and connections to related concepts

**Writing Style**: Use formal, academic textbook style. Present information objectively and authoritatively. NEVER reference the speaker, author, or video. State all information as established facts and concepts."""

    def _inject_timestamps(self, word_timestamps: List[Dict], interval: int = 30) -> str:
        """
        Reconstructs transcript with timestamp markers inserted every 'interval' seconds.
        """
        if not word_timestamps:
            return ""
            
        text_parts = []
        last_marker_time = -interval  # Force marker at start (0s)
        
        for word_data in word_timestamps:
            word = word_data.get("word", "").strip()
            start = word_data.get("start", 0)
            
            # Insert marker if interval has passed
            if start - last_marker_time >= interval:
                text_parts.append(f" [Time: {int(start)}s] ")
                last_marker_time = start
            
            text_parts.append(word)
            
        return " ".join(text_parts)

    def _create_inline_prompt(self, transcript: str, video_title: str, word_timestamps: List[Dict], density: str, detail_level: str) -> str:
        """Build the LLM prompt for inline screenshot placement.

        Injects word-level timestamp markers into the transcript so the LLM
        can reference precise times when placing screenshots.

        Args:
            transcript: Raw transcript text.
            video_title: Title of the video.
            word_timestamps: Word-level timestamp dicts from audio
                transcription.
            density: Screenshot density setting.
            detail_level: Note verbosity setting.

        Returns:
            Fully-assembled prompt string.
        """
        
        # Inject timestamps into the transcript for the LLM to reference
        # This is much better than a separate list which gets truncated
        # Decreased interval to 30s to improve timestamp accuracy
        timestamped_transcript = self._inject_timestamps(word_timestamps, interval=15)
        
        density_instr = self._get_density_instruction(density)
        detail_instr = self._get_detail_instruction(detail_level)
        
        return f"""
        You are an expert video summarizer. Analyze the following transcript for the video titled "{video_title}".
        
        The transcript includes embedded timestamp markers like **[Time: 120s]**. Use these to determine exactly when topics are discussed.
        
        Your goal is to create notes based on the following detail level:
        {detail_instr}
        
        CRITICAL: You must place screenshots **inline** with the text, not at the end of sections.
        - **INTERLEAVE TEXT AND IMAGES**: Do NOT list multiple screenshots in a row. Always provide context text before or after every screenshot.
        - **DESCRIPTIVE CAPTIONS**: Every screenshot MUST have a descriptive caption explaining what is shown (e.g., "Diagram of the stress-strain curve").
        - **DIVERSE TIMESTAMPS**: Request screenshots at timestamps that are AT LEAST 10 seconds apart. Do NOT request multiple screenshots at very similar timestamps.
        - **VISUAL RELEVANCE**: Only request screenshots when there is a likely **visual change** or **diagram** being discussed. Avoid screenshots of just the speaker talking unless they are showing something.
        - **PRECISE TIMING**: Use the nearest **[Time: Xs]** marker and interpolate carefully. If a topic starts 10 seconds after [Time: 60s], the screenshot should be around 62s.
        - **EVEN COUNT**: Each section MUST have an **even number** of screenshots (2, 4, 6, …) so they pair into a 2-column grid.
        - {density_instr}
        
        Return the output strictly as a JSON object with the following structure:
        {{
            "title": "Refined Title",
            "summary": "A detailed summary of the video",
            "sections": [
                {{
                    "heading": "Section Heading",
                    "content": [
                        {{
                            "type": "text",
                            "text": "Paragraph of notes..."
                        }},
                        {{
                            "type": "screenshot",
                            "timestamp": 45.2,
                            "caption": "Detailed description of the visual..."
                        }},
                        {{
                            "type": "text",
                            "text": "More notes explaining the next step..."
                        }}
                    ]
                }}
            ]
        }}
        
        Transcript with Timestamps:
        {timestamped_transcript[:50000]}
        """
    
    def generate_quick_summary(self, transcript: str, video_title: str) -> Dict:
        """
        Generate a quick summary without screenshots or detailed sections.
        Much faster and cheaper than full analysis.
        
        Args:
            transcript: The video transcript text
            video_title: Title of the video
        
        Returns:
            Dict with title and summary only
        """
        prompt = f"""
        You are an expert video summarizer. Create a concise summary for the video titled "{video_title}".
        
        Provide:
        1. **Refined Title**: Improve the title if needed (or keep original if it's already good)
        2. **Comprehensive Summary**: Write a 2-3 paragraph summary covering:
           - Main topic and purpose of the video
           - Key points and takeaways
           - Important concepts or techniques discussed
        
        Return strictly as JSON:
        {{
            "title": "Refined Title Here",
            "summary": "2-3 paragraph comprehensive summary here..."
        }}
        
        Transcript (first 15000 characters):
        {transcript[:15000]}
        """
        
        result = self.provider.generate_json(prompt, max_tokens=4096)
        
        if result:
            return result
        
        return {
            "title": video_title,
            "summary": "Error generating summary. Please try again."
        }
    
    def _create_section_prompt(self, transcript: str, video_title: str, density: str, detail_level: str) -> str:
        """Build the fallback prompt when word-level timestamps are unavailable.

        The LLM is asked to estimate timestamps based on transcript flow
        rather than precise word timings.

        Args:
            transcript: Raw transcript text.
            video_title: Title of the video.
            density: Screenshot density setting.
            detail_level: Note verbosity setting.

        Returns:
            Fully-assembled prompt string.
        """
        
        density_instr = self._get_density_instruction(density)
        detail_instr = self._get_detail_instruction(detail_level)

        return f"""
        You are an expert video summarizer. Analyze the following transcript for the video titled "{video_title}".
        
        Your goal is to create notes based on the following detail level:
        {detail_instr}
        
        CRITICAL: You must place screenshots **inline** with the text, not at the end of sections.
        - **INTERLEAVE TEXT AND IMAGES**: Do NOT list multiple screenshots in a row. Always provide context text before or after every screenshot.
        - **DESCRIPTIVE CAPTIONS**: Every screenshot MUST have a descriptive caption explaining what is shown.
        - **ESTIMATE TIMESTAMPS**: Since you don't have precise word timestamps, estimate the approximate time (in seconds) based on the flow of the transcript (assuming average speaking rate).
        - **EVEN COUNT**: Each section MUST have an **even number** of screenshots (2, 4, 6, …) so they pair into a 2-column grid.
        - {density_instr}
        
        Return the output strictly as a JSON object with the following structure:
        {{
            "title": "Refined Title",
            "summary": "A detailed summary of the video",
            "sections": [
                {{
                    "heading": "Section Heading",
                    "content": [
                        {{
                            "type": "text",
                            "text": "Paragraph of notes..."
                        }},
                        {{
                            "type": "screenshot",
                            "timestamp": 45.5,
                            "caption": "Description of visual..."
                        }},
                        {{
                            "type": "text",
                            "text": "More notes..."
                        }}
                    ]
                }}
            ]
        }}

        Transcript:
        {transcript[:30000]}
        """

if __name__ == "__main__":
    # Test
    pass
