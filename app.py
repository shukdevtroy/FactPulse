import os
import gradio as gr
from groq import Groq
import json
from datetime import datetime
import time
import re

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
            return """
            <div class="tool-info-card">
                <div class="tool-info-header">
                    <i class="icon">🧠</i>
                    <h3>Knowledge Source</h3>
                </div>
                <div class="tool-info-content">
                    <p>Response generated from existing knowledge base</p>
                </div>
            </div>
            """
        
        tool_html = """
        <div class="tool-info-card">
            <div class="tool-info-header">
                <i class="icon">🔍</i>
                <h3>Tools Executed</h3>
            </div>
            <div class="tool-info-content">
        """
        
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
                
                tool_html += f"""
                <div class="tool-item">
                    <div class="tool-name">{i}. {tool_name}</div>
                """
                
                # Add tool parameters if available
                if hasattr(tool, 'parameters'):
                    params = tool.parameters
                    if isinstance(params, dict):
                        for key, value in params.items():
                            tool_html += f'<div class="tool-param">{key}: {value}</div>'
                elif hasattr(tool, 'input'):
                    tool_html += f'<div class="tool-param">Input: {tool.input}</div>'
                    
                tool_html += "</div>"
                    
            except Exception as e:
                tool_html += f'<div class="tool-item"><div class="tool-name">{i}. Tool {i}</div><div class="tool-param">Error parsing details</div></div>'
        
        tool_html += """
            </div>
        </div>
        """
        
        return tool_html
    
    def format_response(self, response_content, query, response_time):
        """Format the response with proper HTML structure"""
        # Convert markdown-like formatting to HTML
        formatted_content = response_content
        
        # Convert **bold** to <strong>
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
        
        # Convert *italic* to <em>
        formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
        
        # Convert newlines to <br> for better formatting
        formatted_content = formatted_content.replace('\n', '<br>')
        
        # Convert numbered lists
        formatted_content = re.sub(r'^(\d+\.\s)', r'<br><strong>\1</strong>', formatted_content, flags=re.MULTILINE)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_response = f"""
        <div class="response-card">
            <div class="response-header">
                <div class="query-section">
                    <h3>📝 Query</h3>
                    <p class="query-text">{query}</p>
                </div>
                <div class="meta-info">
                    <span class="timestamp">🕐 {timestamp}</span>
                    <span class="response-time">⚡ {response_time}s</span>
                </div>
            </div>
            <div class="response-content">
                <h3>💬 Response</h3>
                <div class="response-text">
                    {formatted_content}
                </div>
            </div>
        </div>
        """
        
        return html_response
    
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
    
    # Modern CSS design
    custom_css = """
    /* Reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f8fafc;
        color: #334155;
        line-height: 1.6;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
        padding: 20px;
        background: #f8fafc;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .app-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    .card h3 {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicators */
    .status-success {
        background: #10b981;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .status-warning {
        background: #f59e0b;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .status-error {
        background: #ef4444;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Example buttons */
    .example-btn {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #475569;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 0.25rem;
        font-size: 0.875rem;
        text-align: left;
    }
    
    .example-btn:hover {
        background: #e2e8f0;
        border-color: #3b82f6;
        color: #1e40af;
    }
    
    /* Custom prompt examples */
    .prompt-example {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .prompt-example:hover {
        background: #e2e8f0;
        border-color: #3b82f6;
    }
    
    .prompt-example-title {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    .prompt-example-text {
        font-size: 0.875rem;
        color: #64748b;
        line-height: 1.4;
    }
    
    /* Response card */
    .response-card {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .response-header {
        background: #f8fafc;
        padding: 1.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .query-section h3 {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .query-text {
        background: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        font-style: italic;
        color: #475569;
    }
    
    .meta-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
        font-size: 0.875rem;
        color: #64748b;
    }
    
    .timestamp, .response-time {
        background: #f1f5f9;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .response-content {
        padding: 1.5rem;
    }
    
    .response-content h3 {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .response-text {
        color: #374151;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    .response-text strong {
        color: #1e293b;
        font-weight: 600;
    }
    
    .response-text em {
        color: #6366f1;
        font-style: italic;
    }
    
    /* Tool info card */
    .tool-info-card {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .tool-info-header {
        background: #f0f9ff;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .tool-info-header .icon {
        font-size: 1.5rem;
    }
    
    .tool-info-header h3 {
        color: #1e40af;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .tool-info-content {
        padding: 1.5rem;
    }
    
    .tool-item {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    
    .tool-item:last-child {
        margin-bottom: 0;
    }
    
    .tool-name {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .tool-param {
        font-size: 0.875rem;
        color: #64748b;
        margin-left: 1rem;
    }
    
    /* Accordion styles */
    .accordion-header {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 0.5rem;
    }
    
    .accordion-header:hover {
        background: #e2e8f0;
    }
    
    .accordion-content {
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        background: white;
    }
    
    /* Performance badge */
    .performance-badge {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-block;
    }
    
    /* Buttons */
    .btn-primary {
        background: #3b82f6;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .btn-primary:hover {
        background: #2563eb;
    }
    
    .btn-secondary {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .btn-secondary:hover {
        background: #e2e8f0;
        border-color: #3b82f6;
    }
    
    /* Footer */
    .footer {
        background: #1e293b;
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        text-align: center;
    }
    
    .footer h3 {
        color: white;
        margin-bottom: 1rem;
    }
    
    .footer a {
        color: #60a5fa;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        color: #93c5fd;
        text-decoration: underline;
    }
    
    .footer ul {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }
    
    .footer ul li {
        margin: 0.5rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 1rem;
        }
        
        .app-header h1 {
            font-size: 2rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .meta-info {
            flex-direction: column;
            gap: 0.5rem;
        }
    }
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
        
        # Format response with HTML
        formatted_response = fact_checker.format_response(response, query, response_time)
        
        return formatted_response, tool_info or "", f"⚡ Response time: {response_time}s"
    
    def reset_system_prompt():
        return fact_checker.get_system_prompt()
    
    def load_example(example_text):
        return example_text
    
    def load_custom_prompt(prompt_text):
        return prompt_text
    
    # Create the Gradio interface
    with gr.Blocks(title="Real-time Fact Checker & News Agent", css=custom_css, theme=gr.themes.Ocean()) as demo:
        
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🔍 Real-time Fact Checker & News Agent</h1>
            <p>Powered by Groq's Compound Models with Built-in Web Search</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # API Key section
                gr.HTML('<div class="card">')
                gr.HTML('<h3>🔑 API Configuration</h3>')
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
                gr.HTML('<div class="card">')
                gr.HTML('<h3>⚙️ Advanced Options</h3>')
                
                # Custom System Prompt Examples
                with gr.Accordion("📝 System Prompt Examples", open=False):
                    gr.Markdown("**Click any example to load it as your system prompt:**")
                    
                    custom_prompts = fact_checker.get_custom_prompt_examples()
                    for title, prompt in custom_prompts.items():
                        with gr.Row():
                            prompt_btn = gr.Button(title, variant="secondary", size="sm")
                            prompt_btn.click(
                                fn=lambda p=prompt: p,
                                outputs=[gr.Textbox(elem_id="system_prompt_input")]
                            )
                
                with gr.Accordion("🔧 System Prompt Customization", open=False):
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
                gr.HTML('<div class="card">')
                gr.HTML('<h3>💭 Your Query</h3>')
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
                # Example queries
                gr.HTML('<div class="card">')
                gr.HTML('<h3>📝 Example Queries</h3>')
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
        gr.HTML('<h3 style="margin: 2rem 0 1rem 0; color: #1e293b; font-size: 1.5rem;">📊 Results</h3>')
        
        with gr.Row():
            with gr.Column(scale=2):
                response_output = gr.HTML(
                    value="<div style='padding: 2rem; text-align: center; color: #64748b; font-style: italic;'>Your response will appear here...</div>"
                )
            
            with gr.Column(scale=1):
                tool_info_output = gr.HTML(
                    value="<div style='padding: 2rem; text-align: center; color: #64748b; font-style: italic;'>Tool execution details will appear here...</div>"
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
            fn=lambda: ("", "<div style='padding: 2rem; text-align: center; color: #64748b; font-style: italic;'>Your response will appear here...</div>", "<div style='padding: 2rem; text-align: center; color: #64748b; font-style: italic;'>Tool execution details will appear here...</div>", ""),
            outputs=[query_input, response_output, tool_info_output, performance_output]
        )
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <h3>🔗 Useful Links</h3>
            <p>
                <a href="https://console.groq.com/" target="_blank">Groq Console</a> - Get your free API key<br>
                <a href="https://console.groq.com/docs/quickstart" target="_blank">Groq Documentation</a> - Learn more about Groq models<br>
                <a href="https://console.groq.com/docs/models" target="_blank">Compound Models Info</a> - Details about compound models
            </p>
            
            <h3>💡 Tips</h3>
            <ul>
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