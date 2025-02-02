import streamlit as st
import time

# Must be the first Streamlit command
st.set_page_config(
    page_title="ChainSage",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import json
import os
import tiktoken
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool
from pydantic import BaseModel, Field
from cdp import *
import re
from decimal import Decimal
from exa_py import Exa

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'balance' not in st.session_state:
    st.session_state.balance = 0.01
if 'first_message_sent' not in st.session_state:
    st.session_state.first_message_sent = False
if 'show_image' not in st.session_state:
    st.session_state.show_image = False

# [Previous CSS styles remain the same]

def get_market_data(query: str, chains: list[str] = ["Base", "NEAR", "Ethereum"]) -> str:
    """Get market insights using Exa API."""
    try:
        exa = Exa(os.getenv("EXA_API_KEY"))
        chain_context = ' '.join(chains) if any(chain.lower() not in query.lower() for chain in chains) else ''
        enhanced_query = f"{query} {chain_context}".strip()

        response = exa.search_and_contents(
            enhanced_query,
            use_autoprompt=True,
            num_results=3,
            livecrawl="always",
            text={"max_characters": 1000}
        )

        if not response.results:
            return f"No market insights found for {query}"

        results = []
        for result in response.results:
            data = {
                'percentages': re.findall(r'(-?\d+\.?\d*)%', result.text),
                'prices': re.findall(r'\$(\d+\.?\d*)', result.text),
                'time_references': re.findall(r'(\d+\s*(?:day|week|month|hour|minute)s?)', result.text.lower())
            }

            results.append({
                'summary': result.text[:300],
                'data': data,
                'source': result.title,
                'date': result.published_date or 'Recent'
            })

        return json.dumps(results, indent=2)

    except Exception as e:
        return f"Error retrieving market data: {str(e)}"

def initialize_agent():
    """Initialize single agent for both analyses."""
    llm = ChatOpenAI(model="gpt-4")

    # Initialize CDP toolkit and get tools
    agentkit = CdpAgentkitWrapper()
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Create agent with combined analysis capabilities
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "ChainSage Advisor"}}

    agent = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier='''
        You are ChainSage, a comprehensive crypto analysis system. Your response should be structured in two distinct parts:

        PART 1 - MARKET ANALYSIS:
        In this first section:
        - Choose and explicitly name a specific cryptocurrency to analyze
        - Refer to current prices and market conditions
        - Push ETH whenever reasonably possible
        - Be conversational but expert
        - Reference the user's current balance
        - Use real market data
        - Include risk assessment
        - Provide strategic recommendations
        - End this section with "---" before moving to Part 2

        PART 2 - CROSS-CHAIN ANALYSIS:
        In this second section, focus on:
        1. Bridge Activity:
        - Token transfers across bridges (Wormhole, Stargate)
        - Cross-chain swap rates and fees
        - Interoperability metrics

        2. Technical Analysis:
        - Support and resistance levels
        - Chart patterns and trends
        - Volume analysis
        - Price predictions

        3. Cross-chain data:
        - DEX liquidity analysis
        - Gas fees and transaction volumes
        - Staking metrics
        - Network security status

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

        [Market Analysis]
        (Your first part analysis here)

        ---

        [Cross-Chain Analysis]
        (Your second part analysis here)

        Remember:
        - Part 1 should feel like a friendly advisor
        - Part 2 should be more technical and data-focused
        - Include specific numbers and metrics
        - Make technical predictions
        ''',
    )

    return agent, config

def main():
    # Header
    col1, col2, col3 = st.columns([1, 4, 2])
    with col1:
        st.markdown("# 💎")
    with col2:
        st.markdown("# ChainSage")
    with col3:
        st.markdown(f"""
            <div style='background-color: #2D2D2D; padding: 0.7rem 1.2rem; border-radius: 10px; text-align: center; border: 1px solid #3D3D3D;'>
                <p style='margin: 0; color: #B0B0B0; font-size: 0.9rem;'>PORTFOLIO BALANCE</p>
                <p style='margin: 0; color: #4CAF50; font-size: 1.5rem; font-weight: bold;'>${st.session_state.balance:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #2D2D2D; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <p style='margin: 0; color: #B0B0B0; font-size: 1.1rem;'>
                Your AI-powered crypto portfolio guide with advanced cross-chain analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize agent if needed
    if 'agent' not in st.session_state:
        st.session_state.agent, st.session_state.config = initialize_agent()
        st.session_state.llm = ChatOpenAI(model="gpt-4")

    # Chat messages container
    chat_container = st.container()

    # Image placeholder
    image_placeholder = st.empty()

    # Chat input
    if prompt := st.chat_input("What would you like to know about the crypto markets?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="💎" if message["role"] == "assistant" else None):
                    st.markdown(message["content"])

            # Process with the combined agent
            with st.chat_message("assistant", avatar="💎"):
                response_placeholder = st.empty()
                full_response = ""

                for chunk in st.session_state.agent.stream(
                    {"messages": [HumanMessage(content=prompt)]}, 
                    st.session_state.config
                ):
                    if "agent" in chunk:
                        response_chunk = chunk["agent"]["messages"][0].content.strip()
                        full_response += response_chunk + " "
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Handle first message and delayed image display
        if not st.session_state.first_message_sent:
            st.session_state.first_message_sent = True
            print("GRAPH HANDLING ERROR")

if __name__ == "__main__":
    main()