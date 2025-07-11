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
            return True, '<div class="status-success">✅ API Key validated successfully!</div>'
        except Exception as e:
            return False, f'<div class="status-error">❌ Error initializing client: {str(e)}</div>'
    
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
            return '<div class="status-error">❌ Please set a valid API key first.</div>', None, None
        
        try:
            start_time = time.time()
            
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
            
            response_content = chat_completion.choices[0].message.content
            
            executed_tools = getattr(chat_completion.choices[0].message, 'executed_tools', None)
            
            tool_info = self.format_tool_info(executed_tools)
            
            return response_content, tool_info, response_time
            
        except Exception as e:
            return f'<div class="status-error">❌ Error querying model: {str(e)}</div>', None, None
    
    def format_tool_info(self, executed_tools):
        """Format executed tools information for display"""
        if not executed_tools:
            return '<div class="tool-info"><strong>🔍 Tools Used:</strong> None (Used existing knowledge)</div>'
        
        tool_info = '<div class="tool-info"><strong>🔍 Tools Used:</strong><ul>'
        for i, tool in enumerate(executed_tools, 1):
            try:
                if hasattr(tool, 'name'):
                    tool_name = tool.name
                elif hasattr(tool, 'tool_name'):
                    tool_name = tool.tool_name
                elif isinstance(tool, dict):
                    tool_name = tool.get('name', 'Unknown')
                else:
                    tool_name = str(tool)
                
                tool_info += f'<li>{i}. <strong>{tool_name}</strong>'
                
                if hasattr(tool, 'parameters'):
                    params = tool.parameters
                    if isinstance(params, dict):
                        tool_info += '<ul>'
                        for key, value in params.items():
                            tool_info += f'<li>{key}: {value}</li>'
                        tool_info += '</ul>'
                elif hasattr(tool, 'input'):
                    tool_info += f'<ul><li>Input: {tool.input}</li></ul>'
                tool_info += '</li>'
                
            except Exception:
                tool_info += f'<li>{i}. <strong>Tool {i}</strong> (Error parsing details)</li>'
        
        tool_info += '</ul></div>'
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
    
    custom_css = """
    <style>
    .gradio-container {
        max-width: 1400px;
        margin: 0 auto;
        background: #f5f7fa;
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1f2937;
    }
    
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #111827;
    }
    
    .main-header p {
        font-size: 1rem;
        color: #4b5563;
        margin: 0.5rem 0 0;
    }
    
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease;
    }
    
    .section-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .example-item {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e5e7eb;
    }
    
    .example-item:hover {
        background: #e5e7eb;
        transform: translateY(-2px);
    }
    
    .status-success {
        background: #ecfdf5;
        color: #065f46;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #d1fae5;
    }
    
    .status-error {
        background: #fef2f2;
        color: #991b1b;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #fee2e2;
    }
    
    .results-section {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .tool-info {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .performance-badge {
        background: #e5e7eb;
        color: #374151;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        display: inline-block;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    }
    
    .footer-section {
        background: #111827;
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 1rem;
        text-align: center;
    }
    
    .footer-section a {
        color: #60a5fa;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer-section a:hover {
        text-decoration: underline;
    }
    
    .prompt-example {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e5e7eb;
    }
    
    .prompt-example:hover {
        background: #e5e7eb;
    }
    
    .prompt-example-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    
    .prompt-example-text {
        font-size: 0.875rem;
        color: #4b5563;
        line-height: 1.5;
    }
    
    .gr-button {
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    .gr-button-primary {
        background: #2563eb !important;
        color: white !important;
    }
    
    .gr-button-primary:hover {
        background: #1d4ed8 !important;
    }
    
    .gr-button-secondary {
        background: #e5e7eb !important;
        color: #374151 !important;
    }
    
    .gr-button-secondary:hover {
        background: #d1d5db !important;
    }
    
    .gr-textbox, .gr-dropdown, .gr-slider {
        border-radius: 8px !important;
        border: 1px solid #e5e7eb !important;
    }
    
    .gr-accordion {
        border-radius: 8px !important;
        border: 1px solid #e5e7eb !important;
    }
    </style>
    """
    
    def validate_api_key(api_key):
        if not api_key or api_key.strip() == "":
            return '<div class="status-error">❌ Please enter a valid API key</div>', False
        
        success, message = fact_checker.initialize_client(api_key.strip())
        return message, success
    
    def process_query(query, model, temperature, api_key, system_prompt):
        if not api_key or api_key.strip() == "":
            return '<div class="status-error">❌ Please set your API key first</div>', "", ""
        
        if not query or query.strip() == "":
            return '<div class="status-error">❌ Please enter a query</div>', "", ""
        
        if not fact_checker.client:
            success, message = fact_checker.initialize_client(api_key.strip())
            if not success:
                return message, "", ""
        
        response, tool_info, response_time = fact_checker.query_compound_model(
            query.strip(), model, temperature, system_prompt.strip() if system_prompt else None
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_response = f"""
        <div class="results-section">
            <h3>Query</h3>
            <p>{query}</p>
            <h3>Response</h3>
            <div>{response}</div>
            <hr>
            <p><em>Generated at {timestamp} in {response_time}s</em></p>
        </div>
        """
        
        return formatted_response, tool_info or "", f'<div class="performance-badge">⚡ Response time: {response_time}s</div>'
    
    def reset_system_prompt():
        return fact_checker.get_system_prompt()
    
    def load_example(example_text):
        return example_text
    
    def load_custom_prompt(prompt_text):
        return prompt_text
    
    with gr.Blocks(title="Real-time Fact Checker & News Agent", css=custom_css) as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 Real-time Fact Checker</h1>
            <p>Powered by Groq's Compound Models with Real-time Web Search</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.HTML('<div class="section-card">')
                    gr.Markdown("### 🔑 API Configuration")
                    api_key_input = gr.Textbox(
                        label="Groq API Key",
                        placeholder="Enter your Groq API key here...",
                        type="password",
                        info="Get your free API key from https://console.groq.com/"
                    )
                    api_status = gr.HTML(
                        value='<div class="status-error">⚠️ Please enter your API key</div>'
                    )
                    validate_btn = gr.Button("Validate API Key", variant="secondary")
                    gr.HTML('</div>')
                
                with gr.Group():
                    gr.HTML('<div class="section-card">')
                    gr.Markdown("### ⚙️ Advanced Options")
                    
                    with gr.Accordion("📝 System Prompt Examples", open=False):
                        gr.Markdown("**Click any example to load it as your system prompt:**")
                        custom_prompts = fact_checker.get_custom_prompt_examples()
                        for title, prompt in custom_prompts.items():
                            gr.HTML(f"""
                            <div class="prompt-example" onclick="document.getElementById('system_prompt_input').value = '{prompt}'">
                                <div class="prompt-example-title">{title}</div>
                                <div class="prompt-example-text">{prompt[:100]}...</div>
                            </div>
                            """)
                    
                    with gr.Accordion("🔧 System Prompt", open=False):
                        system_prompt_input = gr.Textbox(
                            label="System Prompt",
                            value=fact_checker.get_system_prompt(),
                            lines=8,
                            info="Customize how the AI behaves and responds",
                            elem_id="system_prompt_input"
                        )
                        reset_prompt_btn = gr.Button("Reset to Default", variant="secondary")
                        
                        gr.Markdown("**Quick Load Custom Prompts:**")
                        for title, prompt in custom_prompts.items():
                            prompt_btn = gr.Button(title, variant="secondary")
                            prompt_btn.click(
                                fn=lambda p=prompt: p,
                                outputs=[system_prompt_input]
                            )
                    gr.HTML('</div>')
                
                with gr.Group():
                    gr.HTML('<div class="section-card">')
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
                            label="Model"
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                    
                    submit_btn = gr.Button("🔍 Get Real-time Information", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                    gr.HTML('</div>')
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML('<div class="section-card">')
                    gr.Markdown("### 📝 Example Queries")
                    gr.Markdown("Click any example to load it:")
                    
                    examples = fact_checker.get_example_queries()
                    for category, queries in examples.items():
                        with gr.Accordion(category, open=category == "📰 Latest News"):
                            for query in queries:
                                example_btn = gr.Button(query, variant="secondary")
                                example_btn.click(
                                    fn=lambda q=query: q,
                                    outputs=[query_input]
                                )
                    gr.HTML('</div>')
        
        gr.HTML('<div class="results-section">')
        gr.Markdown("### 📊 Results")
        
        with gr.Row():
            with gr.Column(scale=2):
                response_output = gr.HTML(
                    value='<div class="results-section"><em>Your response will appear here...</em></div>'
                )
            
            with gr.Column(scale=1):
                tool_info_output = gr.HTML(
                    value='<div class="tool-info"><em>Tool execution details will appear here...</em></div>'
                )
                
                performance_output = gr.HTML(
                    value=""
                )
        gr.HTML('</div>')
        
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
            fn=lambda: ("", '<div class="results-section"><em>Your response will appear here...</em></div>', 
                       '<div class="tool-info"><em>Tool execution details will appear here...</em></div>', ""),
            outputs=[query_input, response_output, tool_info_output, performance_output]
        )
        
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
                <li>Compound models use web search for real-time information</li>
                <li>Use temperature 0.1 for factual queries, 0.7-0.9 for creative ones</li>
                <li>compound-beta: more capable | compound-beta-mini: faster</li>
                <li>Customize system prompts for specialized responses</li>
                <li>Check Tool Execution Info for web search usage</li>
            </ul>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True
    )