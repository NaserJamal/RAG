"""
Hierarchical Map-Reduce Summarization using LLM.

Implements a tree-based approach where:
1. Map: Each document is summarized independently (in parallel)
2. Reduce: Summaries are combined in batches of 10, recursively
3. Final: A single executive summary is produced
"""

import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from pathlib import Path
import json
from datetime import datetime


class HierarchicalSummarizer:
    """
    Hierarchical map-reduce summarization for multiple documents.

    Process flow:
    - Level 0 (Map): Summarize each document individually (parallel)
    - Level 1+ (Reduce): Combine summaries in batches of 10 (parallel within level)
    - Continue until single executive summary
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        batch_size: int = 10,
        max_concurrent: int = 5
    ):
        """
        Initialize the hierarchical summarizer.

        Args:
            api_key: LLM API key
            base_url: LLM API base URL
            model: Model name to use
            batch_size: Number of summaries to combine at each reduce step (default: 10)
            max_concurrent: Maximum concurrent LLM calls (default: 5)
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def summarize_document(self, document_text: str, filename: str) -> Dict[str, str]:
        """
        Summarize a single document.

        Args:
            document_text: Full text of the document
            filename: Name of the document file

        Returns:
            Dictionary with summary and metadata
        """
        async with self.semaphore:
            prompt = f"""You are an expert at creating concise, informative summaries.

Summarize the following document in 3-5 sentences. Focus on:
- Main topics and key points
- Important findings or conclusions
- Critical information that should be preserved

Document: {filename}

{document_text}

Provide a clear, concise summary:"""

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert document summarizer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                summary = response.choices[0].message.content.strip()

                return {
                    "filename": filename,
                    "summary": summary,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }

            except Exception as e:
                raise ValueError(f"Error summarizing {filename}: {str(e)}")

    async def combine_summaries(
        self,
        summaries: List[str],
        level: int,
        batch_num: int
    ) -> Dict[str, str]:
        """
        Combine multiple summaries into one.

        Args:
            summaries: List of summary texts to combine
            level: Current level in the hierarchy (for logging)
            batch_num: Batch number at this level (for logging)

        Returns:
            Dictionary with combined summary and metadata
        """
        async with self.semaphore:
            # Format the summaries
            formatted_summaries = "\n\n".join([
                f"Summary {i+1}:\n{summary}"
                for i, summary in enumerate(summaries)
            ])

            prompt = f"""You are synthesizing multiple document summaries into a cohesive overview.

Combine the following {len(summaries)} summaries into a single, coherent summary.

Guidelines:
- Preserve all key information from each summary
- Identify common themes across summaries
- Remove redundancy while maintaining completeness
- Keep the output concise but comprehensive (5-10 sentences)

{formatted_summaries}

Provide a unified summary:"""

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at synthesizing multiple summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )

                combined = response.choices[0].message.content.strip()

                return {
                    "level": level,
                    "batch": batch_num,
                    "summary": combined,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }

            except Exception as e:
                raise ValueError(f"Error combining summaries (level {level}, batch {batch_num}): {str(e)}")

    async def create_executive_summary(self, final_summary: str, total_docs: int) -> str:
        """
        Create the final executive summary with formatting.

        Args:
            final_summary: The combined summary from all documents
            total_docs: Total number of documents processed

        Returns:
            Formatted executive summary
        """
        async with self.semaphore:
            prompt = f"""You are creating a professional executive summary based on {total_docs} documents.

Transform this synthesized content into a polished executive summary:

{final_summary}

Create a well-structured executive summary with:
- A brief overview (2-3 sentences)
- Key findings or themes (bullet points)
- Critical takeaways or conclusions

Make it professional, clear, and actionable:"""

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an executive summary specialist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                raise ValueError(f"Error creating executive summary: {str(e)}")

    async def process_documents(
        self,
        documents: List[Dict[str, str]],
        output_dir: Path
    ) -> Dict:
        """
        Process all documents using hierarchical map-reduce.

        Args:
            documents: List of dicts with 'filename' and 'text' keys
            output_dir: Directory to save intermediate and final outputs

        Returns:
            Dictionary containing executive summary and processing metadata
        """
        start_time = datetime.now()
        total_tokens = 0
        processing_log = []

        print(f"\n{'='*60}")
        print(f"Starting hierarchical summarization of {len(documents)} documents")
        print(f"{'='*60}\n")

        # Level 0: Map phase - summarize each document
        print(f"[Level 0: Map Phase] Summarizing {len(documents)} documents in parallel...")
        map_tasks = [
            self.summarize_document(doc['text'], doc['filename'])
            for doc in documents
        ]
        doc_summaries = await asyncio.gather(*map_tasks)

        map_tokens = sum(s['tokens_used'] for s in doc_summaries)
        total_tokens += map_tokens

        # Save individual summaries
        summaries_file = output_dir / "01_individual_summaries.json"
        summaries_file.write_text(
            json.dumps(doc_summaries, indent=2),
            encoding='utf-8'
        )

        print(f"✓ Completed {len(doc_summaries)} summaries | Tokens: {map_tokens:,}")
        processing_log.append({
            "level": 0,
            "phase": "map",
            "summaries_generated": len(doc_summaries),
            "tokens_used": map_tokens
        })

        # Extract just the summary text for reduce phase
        current_summaries = [s['summary'] for s in doc_summaries]
        level = 1

        # Reduce phase: hierarchically combine summaries
        while len(current_summaries) > 1:
            print(f"\n[Level {level}: Reduce Phase] Combining {len(current_summaries)} summaries...")

            # Split into batches
            batches = [
                current_summaries[i:i + self.batch_size]
                for i in range(0, len(current_summaries), self.batch_size)
            ]

            print(f"  → Processing {len(batches)} batches of up to {self.batch_size} summaries each")

            # Combine each batch in parallel
            combine_tasks = [
                self.combine_summaries(batch, level, batch_num)
                for batch_num, batch in enumerate(batches)
            ]
            batch_results = await asyncio.gather(*combine_tasks)

            level_tokens = sum(r['tokens_used'] for r in batch_results)
            total_tokens += level_tokens

            # Save level results
            level_file = output_dir / f"02_level_{level}_summaries.json"
            level_file.write_text(
                json.dumps(batch_results, indent=2),
                encoding='utf-8'
            )

            print(f"✓ Generated {len(batch_results)} combined summaries | Tokens: {level_tokens:,}")
            processing_log.append({
                "level": level,
                "phase": "reduce",
                "batches_processed": len(batches),
                "summaries_generated": len(batch_results),
                "tokens_used": level_tokens
            })

            # Move to next level
            current_summaries = [r['summary'] for r in batch_results]
            level += 1

        # Final executive summary
        print(f"\n[Final Phase] Creating executive summary...")
        executive_summary = await self.create_executive_summary(
            current_summaries[0],
            len(documents)
        )

        # Save final summary
        final_file = output_dir / "03_executive_summary.txt"
        final_file.write_text(executive_summary, encoding='utf-8')

        # Calculate processing time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Create metadata
        metadata = {
            "total_documents": len(documents),
            "total_tokens_used": total_tokens,
            "processing_time_seconds": duration,
            "levels_processed": level,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "timestamp": start_time.isoformat(),
            "processing_log": processing_log
        }

        # Save metadata
        metadata_file = output_dir / "04_metadata.json"
        metadata_file.write_text(
            json.dumps(metadata, indent=2),
            encoding='utf-8'
        )

        print(f"\n{'='*60}")
        print(f"✓ Summarization complete!")
        print(f"  Documents processed: {len(documents)}")
        print(f"  Total tokens used: {total_tokens:,}")
        print(f"  Processing time: {duration:.2f}s")
        print(f"  Levels in hierarchy: {level}")
        print(f"{'='*60}\n")

        return {
            "executive_summary": executive_summary,
            "metadata": metadata,
            "individual_summaries": doc_summaries
        }
