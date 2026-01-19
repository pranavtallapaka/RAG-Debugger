"""
Streamlit App Entry Point

Main application for the RAG Quality Debugger.
"""

import streamlit as st
import pandas as pd
import numpy as np

from evaluators.relevance import RelevanceEvaluator
from evaluators.redundancy import RedundancyEvaluator
from evaluators.coverage import CoverageEvaluator
from example_data_parsed import ALL_EXAMPLES, EXAMPLE_QUERY, EXAMPLE_CHUNKS

# Warning thresholds
HIGH_REDUNDANCY_THRESHOLD = 0.5  # Warn if redundancy ratio exceeds this
LOW_COVERAGE_THRESHOLD = 0.3     # Warn if coverage score is below this


def parse_chunks(chunks_text: str) -> list[str]:
    """Parse chunks from text input (one per line)."""
    if not chunks_text.strip():
        return []
    return [chunk.strip() for chunk in chunks_text.strip().split("\n") if chunk.strip()]


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Quality Debugger",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG Quality Debugger")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        similarity_threshold = st.slider(
            "Redundancy Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Chunks with similarity above this are flagged as redundant"
        )
        missing_context_threshold = st.slider(
            "Missing Context Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Warning if max query-chunk similarity is below this"
        )
    
    # Input section
    st.header("Input")
    
    # Example data selector
    example_names = ["None (Enter manually)"] + list(ALL_EXAMPLES.keys())
    selected_example = st.selectbox(
        "📋 Load Example Dataset",
        example_names,
        help="Select a pre-configured example dataset to test the evaluators"
    )
    
    # Determine default values based on selection
    if selected_example != "None (Enter manually)":
        example_data = ALL_EXAMPLES[selected_example]
        st.info(f"**{selected_example}**: {example_data['description']}")
        default_query = example_data["query"]
        default_chunks = "\n".join(example_data["chunks"])
    else:
        default_query = ""
        default_chunks = ""
    
    query = st.text_area(
        "Query",
        value=default_query,
        placeholder="Enter your query here...",
        height=100
    )
    
    chunks_text = st.text_area(
        "Retrieved Chunks",
        value=default_chunks,
        placeholder="Enter retrieved chunks, one per line...",
        height=200,
        help="Paste your retrieved chunks, one chunk per line"
    )
    
    # Parse chunks
    chunks = parse_chunks(chunks_text)
    
    # Run evaluation button
    if st.button("🔍 Run Evaluation", type="primary", use_container_width=True):
        if not query:
            st.error("⚠️ Please enter a query.")
            return
        
        if not chunks:
            st.error("⚠️ Please enter at least one chunk.")
            return
        
        # Initialize evaluators
        relevance_eval = RelevanceEvaluator(missing_context_threshold=missing_context_threshold)
        redundancy_eval = RedundancyEvaluator(similarity_threshold=similarity_threshold)
        coverage_eval = CoverageEvaluator()
        
        # Run evaluations
        with st.spinner("Running evaluations..."):
            # Relevance
            relevance_results = relevance_eval.evaluate(query, chunks)
            # Extract similarities from results to avoid recomputing embeddings
            similarities = np.array([r["score"] for r in relevance_results])
            missing_context_warning = relevance_eval.check_missing_context(
                query, chunks, similarities=similarities
            )
            
            # Redundancy
            redundancy_results = redundancy_eval.evaluate(chunks)
            
            # Coverage
            coverage_results = coverage_eval.evaluate(query, chunks)
        
        st.markdown("---")
        
        # Display warnings at the top
        if missing_context_warning:
            st.warning(missing_context_warning)
        
        if redundancy_results["redundancy_ratio"] > HIGH_REDUNDANCY_THRESHOLD:
            st.warning(
                f"⚠️ High redundancy detected: {redundancy_results['redundancy_ratio']:.1%} "
                f"of chunk pairs are redundant (similarity > {similarity_threshold:.2f})"
            )
        
        if coverage_results["coverage_score"] < LOW_COVERAGE_THRESHOLD:
            st.warning(
                f"⚠️ Low coverage score: {coverage_results['coverage_score']:.3f}. "
                f"Retrieved chunks may not adequately cover the query."
            )
        
        st.markdown("---")
        
        # Results section
        st.header("Results")
        
        # Relevance table
        st.subheader("📊 Relevance Scores")
        if relevance_results:
            relevance_df = pd.DataFrame(relevance_results)
            relevance_df.index = range(1, len(relevance_df) + 1)
            relevance_df.index.name = "Rank"
            relevance_df["score"] = relevance_df["score"].round(3)
            
            # Highlight low scores
            def highlight_low_scores(row):
                if row["score"] < missing_context_threshold:
                    return ["background-color: #ffcccc"] * len(row)
                return [""] * len(row)
            
            st.dataframe(
                relevance_df.style.apply(highlight_low_scores, axis=1),
                use_container_width=True
            )
        else:
            st.info("No relevance results.")
        
        st.markdown("---")
        
        # Redundancy table
        st.subheader("🔄 Redundancy Detection")
        st.metric("Redundancy Ratio", f"{redundancy_results['redundancy_ratio']:.1%}")
        
        if redundancy_results["flagged_pairs"]:
            redundancy_df = pd.DataFrame(redundancy_results["flagged_pairs"])
            redundancy_df.index = range(1, len(redundancy_df) + 1)
            redundancy_df.index.name = "Pair #"
            redundancy_df["similarity"] = redundancy_df["similarity"].round(3)
            
            # Highlight high similarity
            def highlight_redundant(row):
                return ["background-color: #ffcccc"] * len(row)
            
            st.dataframe(
                redundancy_df.style.apply(highlight_redundant, axis=1),
                use_container_width=True
            )
        else:
            st.success("✓ No redundant pairs detected.")
        
        st.markdown("---")
        
        # Coverage metrics
        st.subheader("📈 Coverage Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Coverage Score", f"{coverage_results['coverage_score']:.3f}")
        with col2:
            st.metric("Avg Similarity", f"{coverage_results['average_similarity']:.3f}")
        with col3:
            st.metric("Max Similarity", f"{coverage_results['max_similarity']:.3f}")
        with col4:
            st.metric("Min Similarity", f"{coverage_results['min_similarity']:.3f}")


if __name__ == "__main__":
    main()

