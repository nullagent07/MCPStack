from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4o")

async def main():
    async with MultiServerMCPClient({
        "math" : {
            "command": "python",
            # Path to the math_server.py file
            "args": ["/Users/platonsolovianov/repos/ai/MCPStack/test/math_server.py"],
            "transport": "stdio",
        },
        "weather" : {
            # Path to the weather_server.py file
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        }
    }) as client:
        agent = create_react_agent(model, client.get_tools())
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
        print(math_response)
        print(weather_response)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
