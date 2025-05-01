import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

# --- Modified Gemini Service ---
class GeminiService:
    def __init__(self, model=None):
        self.model = model

    def preprocess_claim(self, claim):
        """
        Extracts keywords, identifies synonyms, generates a boolean query attempt,
        and categorizes the claim using Gemini.
        """
        if not self.model:
            logger.error("Gemini model not available for preprocessing.")
            # Fallback with claim as boolean query, extract simple keywords
            simple_keywords = [word for word in claim.split() if len(word) > 3]
            return {"keywords": simple_keywords, "boolean_query": claim, "category": "unknown", "synonyms": []} # Add empty synonyms

        prompt = f"""
        Analyze the following claim for academic research retrieval.
        Claim: "{claim}"

        Tasks:
        1.  **Extract Key Terms:** Identify the 5-7 most important nouns, noun phrases, or technical terms central to the claim.
        2.  **Identify Synonyms/Related Terms:** For the most critical key terms identified above, list 1-2 common synonyms or closely related terms used in academic literature (e.g., for "hair loss", suggest "alopecia"; for "heart attack", suggest "myocardial infarction"). If no common synonym exists, omit it for that term.
        3.  **Generate Boolean Query:** Construct ONE robust search query string suitable for academic databases (like PubMed, CrossRef). Use boolean operators (AND, OR, NOT) and parentheses. **Crucially, incorporate the identified synonyms/related terms using OR where appropriate** to broaden the search and capture variations in terminology. Connect the main concepts with AND.
        4.  **Categorize Claim:** Classify the claim into ONE primary category (Health & Medicine, Biology, Physical Sciences, Earth & Env., Technology, Social Sciences, Humanities, Math/CS, General/Other).

        Return ONLY a JSON object with keys "keywords" (list of strings), "synonyms" (list of strings, can be empty), "boolean_query" (single string), and "category" (single string). No explanations.

        Example 1:
        Claim: "Does stress cause baldness?"
        {{
            "keywords": ["stress", "baldness", "cause", "hair"],
            "synonyms": ["alopecia", "hair loss"],
            "boolean_query": "stress AND (baldness OR alopecia OR \\"hair loss\\")",
            "category": "Health & Medicine"
        }}

        Example 2:
        Claim: "Does regular exercise reduce the risk of cardiovascular disease, especially in older adults?"
        {{
            "keywords": ["regular exercise", "cardiovascular disease", "risk reduction", "heart health", "physical activity", "older adults"],
            "synonyms": ["physical activity", "heart disease", "elderly"],
            "boolean_query": "(regular exercise OR \\"physical activity\\") AND (\\"cardiovascular disease\\" OR \\"heart disease\\") AND risk AND (\\"older adults\\" OR elderly)",
            "category": "Health & Medicine"
        }}

        Now, analyze the claim provided above.
        """

        try:
            logger.info(f"Sending enhanced preprocessing request to Gemini for claim: '{claim}'")
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Robust JSON extraction
            try:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                # Validate expected keys (including synonyms, which might be empty)
                if ("keywords" in result and "boolean_query" in result and
                    "category" in result and "synonyms" in result and
                    isinstance(result["keywords"], list) and isinstance(result["synonyms"], list)):

                    logger.info(f"Successfully preprocessed claim. Boolean Query: {result['boolean_query']}, Keywords: {result['keywords']}, Synonyms: {result['synonyms']}, Category: {result['category']}")
                    # Ensure boolean_query is not empty, fallback if needed
                    if not result["boolean_query"].strip():
                        logger.warning("Gemini returned an empty boolean_query, falling back to joined keywords.")
                        # Fallback query construction could also try using synonyms if keywords are empty
                        fallback_terms = result["keywords"] + result["synonyms"]
                        if fallback_terms:
                             result["boolean_query"] = " AND ".join(f'\\"{{term}}\\"' for term in fallback_terms if term) # Add quotes for phrases
                        else:
                             result["boolean_query"] = claim # Last resort
                    # Ensure synonyms key exists even if empty
                    if "synonyms" not in result:
                        result["synonyms"] = []
                    return result
                else:
                    raise ValueError("Missing or invalid keys/types in JSON response.")
            except (ValueError, IndexError, json.JSONDecodeError) as e: # Added JSONDecodeError
                 logger.error(f"Failed to parse JSON from Gemini preprocessing response: {e}. Response: {response_text}")
                 # Fallback: Use claim itself as boolean query, extract simple keywords if possible
                 simple_keywords = [word for word in claim.split() if len(word) > 3] # Basic fallback
                 return {"keywords": simple_keywords, "boolean_query": claim, "category": "unknown", "synonyms": [], "error": "LLM parsing failed"} # Add synonyms

        except Exception as e:
            logger.error(f"Error during Gemini claim preprocessing: {e}")
            # Fallback with claim as boolean query
            simple_keywords = [word for word in claim.split() if len(word) > 3]
            return {"keywords": simple_keywords, "boolean_query": claim, "category": "unknown", "synonyms": [], "error": str(e)} # Add synonyms

    def _generate_rag_synthesis(self, claim: str, evidence_chunks: list[str]) -> dict | None:
        """Internal: First RAG step - Generate detailed and simplified reasoning.

        Args:
            claim: The user's claim.
            evidence_chunks: List of relevant abstract text chunks.

        Returns:
            A dictionary with 'detailed_reasoning' and 'simplified_reasoning', or None on failure.
        """
        if not self.model:
            logger.error("Gemini model not available for RAG synthesis.")
            return None
        if not evidence_chunks:
            logger.warning("No evidence chunks provided for RAG synthesis.")
            return {
                "detailed_reasoning": "No relevant evidence found to analyze the claim.",
                "simplified_reasoning": "No relevant evidence found."
            }

        formatted_evidence = "\n\n".join([f"Evidence Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(evidence_chunks)])

        prompt = f"""
        Analyze the following claim based *only* on the provided evidence chunks extracted from academic abstracts. Your task is to synthesize the findings.

        Claim: "{claim}"

        Evidence Chunks:
        ---
        {formatted_evidence}
        ---

        Instructions:
        1.  Carefully read the claim and each evidence chunk.
        2.  **Distinguish Core Subject vs. External Factors:** Clearly differentiate between evidence directly addressing the properties or effects of the claim's *core subject* versus evidence related to *external factors* (e.g., specific populations, interactions, study limitations mentioned in the evidence).
        3.  Create TWO different summaries:
            a. First, provide a DETAILED SCIENTIFIC summary (3-5 sentences) that references specific evidence chunks using the `[EVIDENCE_CHUNK:NUMBERS]` format (e.g., `[EVIDENCE_CHUNK:5,12,18]`). **Crucially, explicitly mention the distinction identified in step 2 if applicable.** State if the evidence supports/refutes the core subject itself, but external factors introduce caveats (or vice-versa).
            b. Second, provide a SIMPLIFIED summary (2-3 sentences) in plain language. **This summary should also reflect the core subject vs. external factor distinction clearly** but without technical jargon or specific chunk references.

        Return ONLY a JSON object with the keys "detailed_reasoning" and "simplified_reasoning". Do not include any other text, markdown formatting, or explanations outside the JSON structure.

        Example Output Structure:
        {{
            "detailed_reasoning": "Whey protein itself shows promise for muscle synthesis [EVIDENCE_CHUNK:3,7], but concerns about supplement contamination [EVIDENCE_CHUNK:2,18] and interactions [EVIDENCE_CHUNK:62] (external factors) complicate assessment. Effects vary by population [EVIDENCE_CHUNK:11].",
            "simplified_reasoning": "Whey protein might help muscles, but supplements can have contamination issues or side effects. It doesn't work the same for everyone."
        }}

        Now, analyze the claim and evidence provided above.
        """
        try:
            logger.info(f"Sending RAG synthesis request to Gemini for claim: '{claim}'")
            response = self.model.generate_content(prompt)
            response_text = response.text
            # Robust JSON extraction
            try:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                if "detailed_reasoning" in result and "simplified_reasoning" in result:
                    logger.info("Successfully generated RAG synthesis.")
                    return result
                else:
                    raise ValueError("Missing reasoning keys in JSON response.")
            except (ValueError, IndexError, json.JSONDecodeError) as e:
                 logger.error(f"Failed to parse JSON from Gemini RAG synthesis response: {e}. Response: {response_text}")
                 return None # Indicate failure
        except Exception as e:
            logger.error(f"Error during Gemini RAG synthesis: {e}")
            return None # Indicate failure

    def _evaluate_rag_synthesis(self, claim: str, detailed_reasoning: str) -> dict | None:
        """Internal: Second RAG step - Evaluate synthesis and assign verdict/score.

        Args:
            claim: The user's claim.
            detailed_reasoning: The detailed synthesis generated by the first step.

        Returns:
            A dictionary with 'verdict' and 'accuracy_score', or None on failure.
        """
        if not self.model:
            logger.error("Gemini model not available for RAG evaluation.")
            return None
        if not detailed_reasoning:
            logger.error("Detailed reasoning not provided for RAG evaluation.")
            return {"verdict": "Inconclusive", "accuracy_score": 0.0}

        prompt = f"""
        You are a meticulous fact-checking analyst. Evaluate the claim based *only* on the provided Detailed Scientific Summary (which was synthesized from underlying evidence).

        Claim: "{claim}"

        Detailed Scientific Summary:
        --- Start Summary ---
        {detailed_reasoning}
        --- End Summary ---

        Instructions:
        1.  Read the Claim and the Detailed Scientific Summary carefully.
        2.  **Assign a Verdict:** Choose the most appropriate verdict from: "Supported", "Partially Supported", "Refuted", or "Inconclusive" based *only* on the summary.
        3.  **Assign an Accuracy Score:** Provide a numerical score between 0.0 (completely inaccurate) and 1.0 (completely accurate). 
            *   This score should primarily reflect how well the summary supports the **core assertion** or central relationship described in the claim.
            *   Consider the **strength and consistency** of the evidence mentioned for the core assertion.
            *   If the summary indicates strong support for the core assertion but mentions **caveats, conditions, or external factors** (like specific populations, dosage limits, methodology issues), the score can still be high (e.g., 0.7-0.95), reflecting confidence in the core mechanism *under certain circumstances*. The summary *already* explains these limitations.
            *   A score closer to 1.0 implies the core assertion is very broadly and strongly supported with minimal significant caveats mentioned in the summary.
            *   A lower score (e.g., 0.1-0.4) implies the summary indicates the core assertion is weakly supported, contradicted, or evidence is very mixed.
            *   A score around 0.5-0.6 suggests mixed evidence where the core assertion holds sometimes but fails often, or significant caveats strongly limit its applicability as stated in the claim.
            *   Base the score on the **overall weight of evidence for the core assertion** as presented in the summary.

        Return ONLY a JSON object with the keys "verdict" (string) and "accuracy_score" (float). Do not include any other text or explanations.

        Example Output Structure:
        {{
            "verdict": "Partially Supported",
            "accuracy_score": 0.75
        }}

        Now, evaluate the claim based on the detailed summary provided above.
        """
        try:
            logger.info(f"Sending RAG evaluation request to Gemini for claim: '{claim}'")
            response = self.model.generate_content(prompt)
            response_text = response.text
            # Robust JSON extraction
            try:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                # Basic validation
                if "verdict" in result and "accuracy_score" in result and isinstance(result["accuracy_score"], (int, float)):
                    # Clamp score just in case
                    result["accuracy_score"] = max(0.0, min(1.0, float(result["accuracy_score"])))
                    logger.info(f"Successfully evaluated RAG synthesis. Verdict: {result['verdict']}, Score: {result['accuracy_score']}")
                    return result
                else:
                    raise ValueError("Missing or invalid keys/types in JSON response.")
            except (ValueError, IndexError, json.JSONDecodeError) as e:
                 logger.error(f"Failed to parse JSON from Gemini RAG evaluation response: {e}. Response: {response_text}")
                 return None # Indicate failure
        except Exception as e:
            logger.error(f"Error during Gemini RAG evaluation: {e}")
            return None # Indicate failure

    def analyze_with_rag(self, claim, evidence_chunks):
        """Orchestrates the two-step RAG analysis: Synthesis then Evaluation."""
        # Default values in case of errors
        analysis_result = {
            "verdict": "Error",
            "detailed_reasoning": "Analysis failed during synthesis step.",
            "simplified_reasoning": "Analysis failed.",
            "accuracy_score": 0.0
        }

        # Step 1: Generate Synthesis
        synthesis_data = self._generate_rag_synthesis(claim, evidence_chunks)

        if synthesis_data:
            analysis_result["detailed_reasoning"] = synthesis_data["detailed_reasoning"]
            analysis_result["simplified_reasoning"] = synthesis_data["simplified_reasoning"]

            # Step 2: Evaluate Synthesis
            evaluation_data = self._evaluate_rag_synthesis(claim, synthesis_data["detailed_reasoning"])

            if evaluation_data:
                analysis_result["verdict"] = evaluation_data["verdict"]
                analysis_result["accuracy_score"] = evaluation_data["accuracy_score"]
            else:
                # Keep synthesis, but mark evaluation failure
                analysis_result["verdict"] = "Error"
                analysis_result["detailed_reasoning"] += " \n[Evaluation step failed]"
                analysis_result["accuracy_score"] = 0.0
        else:
            # Synthesis failed, use default error messages already set
            pass

        # Log the final outcome before returning
        logger.info(f"Final RAG Analysis Result - Verdict: {analysis_result['verdict']}, Score: {analysis_result['accuracy_score']}")
        return analysis_result

# --- End Gemini Service --- 