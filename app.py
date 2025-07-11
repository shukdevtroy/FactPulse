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
            return True, "<div class='status-success'>✅ API Key validated successfully!</div>"
        except Exception as e:
            return False, f"<div class='status-error'>❌ Error initializing client: {str(e)}</div>"
    
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
            return "<div class='status-error'>❌ Please set a valid API key first.</div>", None, None
        
        try:
            start_time = time.time()
            
            system_prompt = custom_system_prompt if custom_system_prompt else self.get_system_prompt()
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
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
            return f"<div class='status-error'>❌ Error querying model: {str(e)}</div>", None, None
    
    def format_tool_info(self, executed_tools):
        """Format executed tools information as HTML"""
        if not executed_tools:
            return "<div class='tool-info'>🔍 <strong>Tools Used:</strong> None (Used existing knowledge)</div>"
        
        tool_html = "<div class='tool-info'>🔍 <strong>Tools Used:</strong><ul>"
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
                
                tool_html += f"<li>{i}. <strong>{tool_name}</strong>"
                
                if hasattr(tool, 'parameters'):
                    params = tool.parameters
                    if isinstance(params, dict):
                        tool_html += "<ul>"
                        for key, value in params.items():
                            tool_html += f"<li>{key}: {value}</li>"
                        tool_html += "</ul>"
                elif hasattr(tool, 'input'):
                    tool_html += f"<ul><li>Input: {tool.input}</li></ul>"
                tool_html += "</li>"
            except Exception:
                tool_html += f"<li>{i}. <strong>Tool {i}</strong> (Error parsing details)</li>"
        tool_html += "</ul></div>"
        return tool_html
    
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
            "🎯 Fact-Checker": "You are a fact-checker. Always verify claims with multiple sources and clearly indicate confidence levels in your assessments.",
            "📊 News Analyst": "You are a news analyst. Focus on providing balanced, unbiased reporting with multiple perspectives on current events.",
            "💼 Financial Advisor": "You are a financial advisor. Provide accurate market data with context about trends and implications for investors.",
            "🔬 Research Assistant": "You are a research assistant specializing in scientific and technical information. Provide detailed, evidence-based responses.",
            "🌍 Global News Correspondent": "You are a global news correspondent. Focus on international events and their interconnections.",
            "📈 Market Analyst": "You are a market analyst. Provide detailed financial analysis including technical indicators and market sentiment."
        }

def create_interface():
    fact_checker = RealTimeFactChecker()
    
    custom_css = """
    <style>
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        min-height: 100vh;
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }

    h1, h2, h3, h4 {
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
    }

    .status-success {
        background: #48bb78;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }

    .status-warning {
        background: #ed8936;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }

    .status-error {
        background: #f56565;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }

    .results-section {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .response-content {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .response-content h3 {
        color: #2d3748;
        margin-bottom: 10px;
    }

    .response-content p {
        line-height: 1.6;
        color: #4a5568;
    }

    .timestamp {
        font-size: 0.9rem;
        color: #718096;
        margin-top: 10px;
    }

    .tool-info {
        background: #edf2f7;
        padding: 15px;
        border-radius: 8px;
        color: #2d3748;
    }

    .tool-info ul {
        list-style-type: none;
        padding-left: 0;
    }

    .tool-info li {
        margin-bottom: 8px;
    }

    .performance-badge {
        background: #48bb78;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }

    .footer-section {
        background: #2d3748;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }

    .footer-section a {
        color: #63b3ed;
        text-decoration: none;
    }

    .footer-section a:hover {
        text-decoration: underline;
    }

    @media (max-width: 768px) {
        .gradio-row {
            flex-direction: column;
        }
    }
    </style>
    """
    
    def validate_api_key(api_key):
        if not api_key or api_key.strip() == "":
            return "<div class='status-error'>❌ Please enter a valid API key</div>", False
        success, message = fact_checker.initialize_client(api_key.strip())
        return message, success
    
    def process_query(query, model, temperature, api_key, system_prompt):
        if not api_key or api_key.strip() == "":
            return "<div class='status-error'>❌ Please set your API key first</div>", "", ""
        
        if not query or query.strip() == "":
            return "<div class='status-error'>❌ Please enter a query</div>", "", ""
        
        if not fact_checker.client:
            success, message = fact_checker.initialize_client(api_key.strip())
            if not success:
                return message, "", ""
        
        response, tool_info, response_time = fact_checker.query_compound_model(
            query.strip(), model, temperature, system_prompt.strip() if system_prompt else None
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_response = f"""
        <div class='response-content'>
            <h3>Query: {query}</h3>
            <p>{response}</p>
            <p class='timestamp'>Generated at {timestamp} in {response_time}s</p>
        </div>
        """
        
        performance_html = f"<span class='performance-badge'>⚡ Response time: {response_time}s</span>"
        
        return formatted_response, tool_info or "", performance_html
    
    def reset_system_prompt():
        return fact_checker.get_system_prompt()
    
    def load_example(example_text):
        return example_text
    
    def load_custom_prompt(prompt_text):
        return prompt_text
    
    with gr.Blocks(title="Real-time Fact Checker & News Agent", css=custom_css) as demo:
        gr.HTML("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🔍 Real-time Fact Checker & News Agent</h1>
            <p style='color: white;'>Powered by Groq's Compound Models with Built-in Web Search</p>
        </div>
        """)
        
        with gr.Row(elem_classes="gradio-row"):
            with gr.Column(scale=2):
                with gr.Group():
                    gr.HTML('<div class="feature-card">')
                    gr.Markdown("### 🔑 API Configuration")
                    api_key_input = gr.Textbox(
                        label="Groq API Key",
                        placeholder="Enter your Groq API key here...",
                        type="password"
                    )
                    api_status = gr.HTML(
                        label="Status",
                        value="<div class='status-warning'>⚠️ Please enter your API key</div>"
                    )
                    validate_btn = gr.Button("Validate API Key", variant="secondary")
                    gr.HTML('</div>')
                
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
                    gr.HTML('<div class="feature-card">')
                    gr.Markdown("### 📝 Example Queries")
                    examples = fact_checker.get_example_queries()
                    for category, queries in examples.items():
                        with gr.Accordion(category, open=False):
                            for query in queries:
                                gr.Button(query, variant="secondary").click(
                                    fn=lambda q=query: q,
                                    outputs=[query_input]
                                )
                    gr.HTML('</div>')
        
        gr.HTML('<div class="results-section">')
        gr.Markdown("### 📊 Results")
        with gr.Row():
            with gr.Column(scale=2):
                response_output = gr.HTML(
                    label="Response",
                    value="<div class='response-content'><p>Your response will appear here...</p></div>"
                )
            with gr.Column(scale=1):
                tool_info_output = gr.HTML(
                    label="Tool Execution Info",
                    value="<div class='tool-info'>Tool execution details will appear here...</div>"
                )
                performance_output = gr.HTML(
                    label="Performance",
                    value=""
                )
        gr.HTML('</div>')
        
        gr.HTML("""
        <div class="footer-section">
            <h3>🔗 Useful Links</h3>
            <p>
                <a href="https://console.groq.com/" target="_blank">Groq Console</a> |
                <a href="https://console.groq.com/docs/quickstart" target="_blank">Docs</a> |
                <a href="https://console.groq.com/docs/models" target="_blank">Models</a>
            </p>
            <h3>💡 Tips</h3>
            <p>Use custom prompts to specialize the AI. Check tool info for web search usage.</p>
        </div>
        """)
        
        validate_btn.click(
            fn=validate_api_key,
            inputs=[api_key_input],
            outputs=[api_status, gr.State()]
        )
        
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, model_choice, temperature, api_key_input, gr.State()],
            outputs=[response_output, tool_info_output, performance_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "<div class='response-content'><p>Your response will appear here...</p></div>", 
                       "<div class='tool-info'>Tool execution details will appear here...</div>", ""),
            outputs=[query_input, response_output, tool_info_output, performance_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)