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

    def query_compound_model(self, query, model, temperature=0.7, custom_system_prompt=None):
        """Query the compound model and return response with tool execution info"""
        if not self.client:
            return "❌ Please set a valid API key first.", None, None
        
        try:
            start_time = time.time()
            
            # Use custom system prompt if provided
            system_prompt = custom_system_prompt if custom_system_prompt else self.get_system_prompt()
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                model=model,
                temperature=temperature,
                max_tokens=1500
            )
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            # Extract response
            response_content = chat_completion.choices[0].message.content
            
            # Check for executed tools - Fixed the error here
            executed_tools = getattr(chat_completion.choices[0].message, 'executed_tools', None)
            
            # Format tool execution info
            tool_info = self.format_tool_info(executed_tools)
            
            return response_content, tool_info, response_time
            
        except Exception as e:
            return f"❌ Error querying model: {str(e)}", None, None
    
    def format_tool_info(self, executed_tools):
        """Format executed tools information for display - FIXED"""
        if not executed_tools:
            return "🔍 **Tools Used:** None (Used existing knowledge)"
        
        tool_info = "🔍 **Tools Used:**\n"
        for i, tool in enumerate(executed_tools, 1):
            try:
                # Handle different tool object types
                if hasattr(tool, 'name'):
                    tool_name = tool.name
                elif hasattr(tool, 'tool_name'):
                    tool_name = tool.tool_name
                elif isinstance(tool, dict):
                    tool_name = tool.get('name', 'Unknown')
                else:
                    tool_name = str(tool)
                
                tool_info += f"{i}. **{tool_name}**\n"
                
                # Add tool parameters if available
                if hasattr(tool, 'parameters'):
                    params = tool.parameters
                    if isinstance(params, dict):
                        for key, value in params.items():
                            tool_info += f"   - {key}: {value}\n"
                elif hasattr(tool, 'input'):
                    tool_info += f"   - Input: {tool.input}\n"
                    
            except Exception as e:
                tool_info += f"{i}. **Tool {i}** (Error parsing details)\n"
        
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
    
    def get_custom_prompt_examples(self):
        """Return custom system prompt examples"""
        return {
            "🎯 Fact-Checker": "You are a fact-checker. Always verify claims with multiple sources and clearly indicate confidence levels in your assessments. Use phrases like 'highly confident', 'moderately confident', or 'requires verification' when presenting information.",
            
            "📊 News Analyst": "You are a news analyst. Focus on providing balanced, unbiased reporting with multiple perspectives on current events. Always present different viewpoints and avoid partisan language.",
            
            "💼 Financial Advisor": "You are a financial advisor. Provide accurate market data with context about trends and implications for investors. Always include disclaimers about market risks and the importance of professional financial advice.",
            
            "🔬 Research Assistant": "You are a research assistant specializing in scientific and technical information. Provide detailed, evidence-based responses with proper context about methodology and limitations of studies.",
            
            "🌍 Global News Correspondent": "You are a global news correspondent. Focus on international events and their interconnections. Provide cultural context and explain how events in one region might affect others.",
            
            "📈 Market Analyst": "You are a market analyst. Provide detailed financial analysis including technical indicators, market sentiment, and economic factors affecting price movements."
        }

def create_interface():
    fact_checker = RealTimeFactChecker()
    
    # Custom CSS for beautiful styling
    custom_css = """
    <style>
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    
    .example-category {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #667eea;
    }
    
    .example-category h4 {
        margin: 0 0 10px 0;
        color: #2d3748;
        font-weight: 600;
    }
    
    .status-success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .results-section {
        background: white;
        border-radius: 15px;
        padding: 30px;
        margin: 30px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .tool-info {
        background: #f7fafc;
        border-left: 4px solid #4299e1;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .performance-badge {
        background: linear-gradient(135deg, #38b2ac 0%, #319795 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
        margin: 10px 0;
    }
    
    .footer-section {
        background: #2d3748;
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-top: 30px;
        text-align: center;
    }
    
    .footer-section a {
        color: #63b3ed;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer-section a:hover {
        color: #90cdf4;
        text-decoration: underline;
    }
    
    .prompt-example {
        background: #ebf8ff;
        border: 1px solid #bee3f8;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .prompt-example:hover {
        background: #bee3f8;
        transform: translateX(5px);
    }
    
    .prompt-example-title {
        font-weight: 600;
        color: #2b6cb0;
        margin-bottom: 5px;
    }
    
    .prompt-example-text {
        font-size: 0.9rem;
        color: #4a5568;
        line-height: 1.4;
    }
    </style>
    """
    
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
        
        response, tool_info, response_time = fact_checker.query_compound_model(
            query.strip(), model, temperature, system_prompt.strip() if system_prompt else None
        )
        
        # Format response with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_response = f"**Query:** {query}\n\n**Response:**\n{response}\n\n---\n*Generated at {timestamp} in {response_time}s*"
        
        return formatted_response, tool_info or "", f"⚡ Response time: {response_time}s"
    
    def reset_system_prompt():
        return fact_checker.get_system_prompt()
    
    def load_example(example_text):
        return example_text
    
    def load_custom_prompt(prompt_text):
        return prompt_text
    
    # Create the Gradio interface
    with gr.Blocks(title="Real-time Fact Checker & News Agent", css=custom_css) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 Real-time Fact Checker & News Agent</h1>
            <p>Powered by Groq's Compound Models with Built-in Web Search</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # API Key section
                with gr.Group():
                    gr.HTML('<div class="feature-card">')
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
                    gr.HTML('</div>')
                
                # Advanced options
                with gr.Group():
                    gr.HTML('<div class="feature-card">')
                    gr.Markdown("### ⚙️ Advanced Options")
                    
                    # Custom System Prompt Examples
                    with gr.Accordion("📝 System Prompt Examples (Click to view)", open=False):
                        gr.Markdown("**Click any example to load it as your system prompt:**")
                        
                        custom_prompts = fact_checker.get_custom_prompt_examples()
                        for title, prompt in custom_prompts.items():
                            with gr.Row():
                                gr.HTML(f"""
                                <div class="prompt-example" onclick="document.getElementById('system_prompt_input').value = '{prompt}'">
                                    <div class="prompt-example-title">{title}</div>
                                    <div class="prompt-example-text">{prompt[:100]}...</div>
                                </div>
                                """)
                    
                    with gr.Accordion("🔧 System Prompt (Click to customize)", open=False):
                        system_prompt_input = gr.Textbox(
                            label="System Prompt",
                            value=fact_checker.get_system_prompt(),
                            lines=8,
                            info="Customize how the AI behaves and responds",
                            elem_id="system_prompt_input"
                        )
                        reset_prompt_btn = gr.Button("Reset to Default", variant="secondary", size="sm")
                        
                        # Add buttons for each custom prompt
                        gr.Markdown("**Quick Load Custom Prompts:**")
                        custom_prompts = fact_checker.get_custom_prompt_examples()
                        for title, prompt in custom_prompts.items():
                            prompt_btn = gr.Button(title, variant="secondary", size="sm")
                            prompt_btn.click(
                                fn=lambda p=prompt: p,
                                outputs=[system_prompt_input]
                            )
                    
                    gr.HTML('</div>')

                # Query section
                with gr.Group():
                    gr.HTML('<div class="feature-card">')
                    gr.Markdown("### 💭 Your Query")
                    query_input = gr.Textbox(
                        label="Ask anything that requires real-time information",
                        placeholder="e.g., What are the latest AI developments today?",
                        lines=4
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
                    gr.HTML('</div>')
            
            with gr.Column(scale=1):
                # Example queries with tabs
                with gr.Group():
                    gr.HTML('<div class="feature-card">')
                    gr.Markdown("### 📝 Example Queries")
                    gr.Markdown("Click any example to load it:")
                    
                    examples = fact_checker.get_example_queries()
                    
                    with gr.Accordion("📰 Latest News", open=True):
                        for query in examples["📰 Latest News"]:
                            example_btn = gr.Button(query, variant="secondary", size="sm")
                            example_btn.click(
                                fn=lambda q=query: q,
                                outputs=[query_input]
                            )
                    
                    with gr.Accordion("💰 Financial Data", open=False):
                        for query in examples["💰 Financial Data"]:
                            example_btn = gr.Button(query, variant="secondary", size="sm")
                            example_btn.click(
                                fn=lambda q=query: q,
                                outputs=[query_input]
                            )
                    
                    with gr.Accordion("🌤️ Weather Updates", open=False):
                        for query in examples["🌤️ Weather Updates"]:
                            example_btn = gr.Button(query, variant="secondary", size="sm")
                            example_btn.click(
                                fn=lambda q=query: q,
                                outputs=[query_input]
                            )
                    
                    with gr.Accordion("🔬 Science & Technology", open=False):
                        for query in examples["🔬 Science & Technology"]:
                            example_btn = gr.Button(query, variant="secondary", size="sm")
                            example_btn.click(
                                fn=lambda q=query: q,
                                outputs=[query_input]
                            )
                    
                    with gr.Accordion("🏆 Sports & Entertainment", open=False):
                        for query in examples["🏆 Sports & Entertainment"]:
                            example_btn = gr.Button(query, variant="secondary", size="sm")
                            example_btn.click(
                                fn=lambda q=query: q,
                                outputs=[query_input]
                            )
                    
                    with gr.Accordion("🔍 Fact Checking", open=False):
                        for query in examples["🔍 Fact Checking"]:
                            example_btn = gr.Button(query, variant="secondary", size="sm")
                            example_btn.click(
                                fn=lambda q=query: q,
                                outputs=[query_input]
                            )
                    
                    gr.HTML('</div>')
        
        # Results section
        gr.HTML('<div class="results-section">')
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
        gr.HTML('</div>')
        
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
        gr.HTML("""
        <div class="footer-section">
            <h3>🔗 Useful Links</h3>
            <p>
                <a href="https://console.groq.com/" target="_blank">Groq Console</a> - Get your free API key<br>
                <a href="https://console.groq.com/docs/quickstart" target="_blank">Groq Documentation</a> - Learn more about Groq models<br>
                <a href="https://console.groq.com/docs/models" target="_blank">Compound Models Info</a> - Details about compound models
            </p>
            
            <h3>💡 Tips</h3>
            <ul style="text-align: left; display: inline-block;">
                <li>The compound models automatically use web search when real-time information is needed</li>
                <li>Try different temperature settings: 0.1 for factual queries, 0.7-0.9 for creative questions</li>
                <li>compound-beta is more capable but slower, compound-beta-mini is faster but less capable</li>
                <li>Use custom system prompts to specialize the AI for different types of queries</li>
                <li>Check the Tool Execution Info to see when web search was used</li>
            </ul>
        </div>
        """)
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True
    )