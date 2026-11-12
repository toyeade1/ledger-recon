"""Real stdio MCP -> running local HTTP console functional check."""
import asyncio,json,sys
from pathlib import Path
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params=StdioServerParameters(command=sys.executable,args=['-m','ledger_recon.investigator.mcp_server','--url','http://127.0.0.1:8789'],cwd=str(Path(__file__).resolve().parents[1]))
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            tools=await session.list_tools();names=sorted(x.name for x in tools.tools)
            assert names==['inspect_payment_case','list_payment_cases','search_payment_evidence'],names
            for name,args in [('list_payment_cases',{}),('inspect_payment_case',{'case_id':'PAY-CREDIT'}),('search_payment_evidence',{'query':'Northstar credit','limit':3})]:
                result=await session.call_tool(name,args)
                assert not result.is_error,(name,result)
    print(json.dumps({'passed':True,'read_only_tools':names,'transport':'stdio to loopback HTTP'},indent=2))
if __name__=='__main__':asyncio.run(main())
