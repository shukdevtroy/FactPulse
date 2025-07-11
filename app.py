import os
import gradio as gr
from groq import Groq
import json
from datetime import datetime
import time

class RealTimeFactChecker:
    def __init__(self):
        self.client = None
        self.model_options = ["compound-beta", "compound-beta-mini"]
        
    def initialize_client(self, api_key):
        """Initialize Groq client with API key"""
        try:
            self.client = Groq(api_key=api_key)
            return True, "✅ API Key validated successfully!"
        except Exception as e:
            return False, f"❌ Error initializing client: {str(e)}"
    
    def get_system_prompt(self):
        """Get the system prompt for consistent behavior"""
        return """You are a Real-time Fact Checker and News Agent. Your primary role is to provide accurate, up-to-date information by leveraging web search when needed.

CORE RESPONSIBILITIES:
1. **Fact Verification**: Always verify claims with current, reliable sources
2. **Real-time Information**: Use web search for any information that changes frequently (news, stocks, weather, current events)
3. **Source Transparency**: When using web search, mention the sources or indicate that you've searched for current information
4. **Accuracy First**: If information is uncertain or conflicting, acknowledge this clearly

RESPONSE GUIDELINES:
- **Structure**: Start with a clear, direct answer, then provide supporting details
- **Recency**: Always prioritize the most recent, reliable information
- **Clarity**: Use clear, professional language while remaining accessible
- **Completeness**: Provide comprehensive answers but stay focused on the query
- **Source Awareness**: When you've searched for information, briefly indicate this (e.g., "Based on current reports..." or "Recent data shows...")

WHEN TO SEARCH:
- Breaking news or current events
- Stock prices, market data, or financial information
- Weather conditions or forecasts
- Recent scientific discoveries or research
- Current political developments
- Real-time statistics or data
- Verification of recent claims or rumors

RESPONSE FORMAT:
- Lead with key facts
- Include relevant context
- Mention timeframe when relevant (e.g., "as of today", "this week")
- If multiple sources conflict, acknowledge this
- End with a clear summary for complex topics

Remember: Your goal is to be the most reliable, up-to-date source of information possible."""

    def query_compound_model(self, query, model, temperature=0.7):
        """Query the compound model and return response with tool execution info"""
        if not self.client:
            return "❌ Please set a valid API key first.", None, None
        
        try:
            start_time = time.time()
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                model=model,
                temperature=temperature,
                max_tokens=1000
            )
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            # Extract response
            response_content = chat_completion.choices[0].message.content
            
            # Check for executed tools
            executed_tools = getattr(chat_completion.choices[0].message, 'executed_tools', None)
            
            # Format tool execution info
            tool_info = self.format_tool_info(executed_tools)
            
            return response_content, tool_info, response_time
            
        except Exception as e:
            return f"❌ Error querying model: {str(e)}", None, None
    
    def format_tool_info(self, executed_tools):
        """Format executed tools information for display"""
        if not executed_tools:
            return "🔍 **Tools Used:** None (Used existing knowledge)"
        
        tool_info = "🔍 **Tools Used:**\n"
        for i, tool in enumerate(executed_tools, 1):
            tool_name = tool.get('name', 'Unknown')
            tool_info += f"{i}. **{tool_name}**\n"
            
            # Add tool parameters if available
            if 'parameters' in tool:
                params = tool['parameters']
                if isinstance(params, dict):
                    for key, value in params.items():
                        tool_info += f"   - {key}: {value}\n"
        
        return tool_info
    
    def get_example_queries(self):
        """Return categorized example queries"""
        return {
            "📰 Latest News": [
                "What are the top 3 news stories today?",
                "Latest developments in AI technology this week",
                "Recent political events in the United States",
                "Breaking news about climate change",
                "What happened in the stock market today?"
            ],
            "💰 Financial Data": [
                "Current price of Bitcoin",
                "Tesla stock price today",
                "How is the S&P 500 performing today?",
                "Latest cryptocurrency market trends",
                "What's the current inflation rate?"
            ],
            "🌤️ Weather Updates": [
                "Current weather in New York City",
                "Weather forecast for London this week",
                "Is it going to rain in San Francisco today?",
                "Temperature in Tokyo right now",
                "Weather conditions in Sydney"
            ],
            "🔬 Science & Technology": [
                "Latest breakthroughs in fusion energy",
                "Recent discoveries in space exploration",
                "New developments in quantum computing",
                "Latest medical research findings",
                "Recent advances in renewable energy"
            ],
            "🏆 Sports & Entertainment": [
                "Latest football match results",
                "Who won the recent tennis tournament?",
                "Box office numbers for this weekend",
                "Latest movie releases this month",
                "Recent celebrity news"
            ],
            "🔍 Fact Checking": [
                "Is it true that the Earth's population reached 8 billion?",
                "Verify: Did company X announce layoffs recently?",
                "Check if the recent earthquake in Turkey was magnitude 7+",
                "Confirm the latest unemployment rate statistics",
                "Verify recent claims about electric vehicle sales"
            ]
        }

def create_interface():
    fact_checker = RealTimeFactChecker()
    
    def validate_api_key(api_key):
        if not api_key or api_key.strip() == "":
            return "❌ Please enter a valid API key", False
        
        success, message = fact_checker.initialize_client(api_key.strip())
        return message, success
    
    def process_query(query, model, temperature, api_key, system_prompt):
        if not api_key or api_key.strip() == "":
            return "❌ Please set your API key first", "", ""
        
        if not query or query.strip() == "":
            return "❌ Please enter a query", "", ""
        
        # Initialize client if not already done
        if not fact_checker.client:
            success, message = fact_checker.initialize_client(api_key.strip())
            if not success:
                return message, "", ""
        
        # Use custom system prompt if provided
        if system_prompt and system_prompt.strip():
            original_prompt = fact_checker.get_system_prompt
            fact_checker.get_system_prompt = lambda: system_prompt.strip()
        
        response, tool_info, response_time = fact_checker.query_compound_model(
            query.strip(), model, temperature
        )
        
        # Restore original system prompt function
        if system_prompt and system_prompt.strip():
            fact_checker.get_system_prompt = original_prompt
        
        # Format response with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_response = f"**Query:** {query}\n\n**Response:**\n{response}\n\n---\n*Generated at {timestamp} in {response_time}s*"
        
        return formatted_response, tool_info or "", f"⚡ Response time: {response_time}s"
    
    def reset_system_prompt():
        return fact_checker.get_system_prompt()
    
    def load_example(example_text):
        return example_text
    
    # Create the Gradio interface
    with gr.Blocks(title="Real-time Fact Checker & News Agent", theme=gr.themes.Ocean()) as demo:
        gr.Markdown("""
        # 🔍 Real-time Fact Checker & News Agent
        
        **Powered by Groq's Compound Models with Built-in Web Search**
        
        This application provides real-time information by automatically searching the web when needed. 
        Enter your query below and get up-to-the-minute facts, news, and data!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # API Key section
                with gr.Group():
                    gr.Markdown("### 🔑 API Configuration")
                    api_key_input = gr.Textbox(
                        label="Groq API Key",
                        placeholder="Enter your Groq API key here...",
                        type="password",
                        info="Get your free API key from https://console.groq.com/"
                    )
                    api_status = gr.Textbox(
                        label="Status",
                        value="⚠️ Please enter your API key",
                        interactive=False
                    )
                    validate_btn = gr.Button("Validate API Key", variant="secondary")
                
                # Advanced options
                with gr.Group():
                    gr.Markdown("### ⚙️ Advanced Options")
                    with gr.Accordion("System Prompt (Click to customize)", open=False):
                        system_prompt_input = gr.Textbox(
                            label="System Prompt",
                            value=fact_checker.get_system_prompt(),
                            lines=8,
                            info="Customize how the AI behaves and responds"
                        )
                        reset_prompt_btn = gr.Button("Reset to Default", variant="secondary", size="sm")

                # Query section
                with gr.Group():
                    gr.Markdown("### 💭 Your Query")
                    query_input = gr.Textbox(
                        label="Ask anything that requires real-time information",
                        placeholder="e.g., What are the latest AI developments today?",
                        lines=3
                    )
                    
                    with gr.Row():
                        model_choice = gr.Dropdown(
                            choices=fact_checker.model_options,
                            value="compound-beta",
                            label="Model",
                            info="compound-beta: More capable | compound-beta-mini: Faster"
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Higher = more creative, Lower = more focused"
                        )
                    
                    submit_btn = gr.Button("🔍 Get Real-time Information", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary")
            
            with gr.Column(scale=1):
                # Example queries
                with gr.Group():
                    gr.Markdown("### 📝 Example Queries")
                    gr.Markdown("Click any example to load it:")
                    
                    examples = fact_checker.get_example_queries()
                    for category, queries in examples.items():
                        gr.Markdown(f"**{category}**")
                        for query in queries:
                            example_btn = gr.Button(query, variant="secondary", size="sm")
                            example_btn.click(
                                fn=load_example,
                                inputs=[gr.State(query)],
                                outputs=[query_input]
                            )
        
        # Results section
        gr.Markdown("### 📊 Results")
        
        with gr.Row():
            with gr.Column(scale=2):
                response_output = gr.Markdown(
                    label="Response",
                    value="*Your response will appear here...*"
                )
            
            with gr.Column(scale=1):
                tool_info_output = gr.Markdown(
                    label="Tool Execution Info",
                    value="*Tool execution details will appear here...*"
                )
                
                performance_output = gr.Textbox(
                    label="Performance",
                    value="",
                    interactive=False
                )
        
        # Event handlers
        validate_btn.click(
            fn=validate_api_key,
            inputs=[api_key_input],
            outputs=[api_status, gr.State()]
        )
        
        reset_prompt_btn.click(
            fn=reset_system_prompt,
            outputs=[system_prompt_input]
        )
        
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, model_choice, temperature, api_key_input, system_prompt_input],
            outputs=[response_output, tool_info_output, performance_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "*Your response will appear here...*", "*Tool execution details will appear here...*", ""),
            outputs=[query_input, response_output, tool_info_output, performance_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🔗 Useful Links
        - [Groq Console](https://console.groq.com/) - Get your free API key
        - [Groq Documentation](https://console.groq.com/docs/quickstart) - Learn more about Groq models
        - [Compound Models Info](https://console.groq.com/docs/models) - Details about compound models
        
        ### 💡 Tips
        - The compound models automatically use web search when real-time information is needed
        - Try different temperature settings: 0.1 for factual queries, 0.7-0.9 for creative questions
        - compound-beta is more capable but slower, compound-beta-mini is faster but less capable
        """)
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True
    )