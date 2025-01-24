#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import subprocess
import textwrap

############################
# 1. Create toy data (data.csv) if it doesn't exist
############################
toy_data = textwrap.dedent("""\
genotype,environment,yield
G1,E1,5.5
G2,E1,6.0
G3,E1,5.8
G1,E2,6.5
G2,E2,7.0
G3,E2,6.7
G1,E3,7.1
G2,E3,7.5
G3,E3,7.2
""")

if not os.path.exists("data.csv"):
    with open("data.csv", "w") as f:
        f.write(toy_data)

############################
# 2. Check for blup_analysis.R
############################
# Make sure your script is in the same directory with the name "blup_analysis.R".
# Although if not found the script create one dyamically
# If needed, you can also verify or create it programmatically.

if not os.path.exists("blup_analysis.R"):
    r_script = textwrap.dedent("""\
    #!/usr/bin/env Rscript
    
    args <- commandArgs(trailingOnly=TRUE)
    if (length(args) < 2) {
      stop("Usage: Rscript blup_analysis.R <input_csv> <output_csv>")
    }
    
    input_csv <- args[1]
    output_csv <- args[2]
    
    if(!require(lme4)) {
      install.packages("lme4", repos='http://cran.us.r-project.org')
      library(lme4)
    }
    
    df <- read.csv(input_csv)
    
    model <- lmer(yield ~ (1 | genotype) + (1 | environment), data = df)
    genotype_blups <- ranef(model)$genotype
    
    write.csv(genotype_blups, output_csv, row.names=TRUE)
    """)
    with open("blup_analysis.R", "w") as f:
        f.write(r_script)

# Make sure the script is executable on Unix-like systems
if os.name == "posix":
    subprocess.run(["chmod", "+x", "blup_analysis.R"])

############################
# 3. LangChain Setup
############################
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate

# Provide your OpenAI API key here, if using the default OpenAI LLM
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
############################
# 4. Define the BLUP Tool
############################
def run_blup_analysis_tool(input_csv: str = "data.csv",
                           output_csv: str = "blup_output.csv") -> str:
    """
    This function calls the R script 'blup_analysis.R' to compute BLUPs.
    """
    try:
        subprocess.run([
            "Rscript",
            "blup_analysis.R",
            input_csv,
            output_csv
        ], check=True)

        # Read the output CSV (genotype-level BLUPs)
        with open(output_csv, "r") as f:
            result = f.read()
        return "BLUP Output:\n" + result
    except Exception as e:
        return f"An error occurred: {str(e)}"

blup_tool = Tool(
    name="BLUPAnalysisTool",
    func=lambda x: run_blup_analysis_tool(),  # ignoring 'x' for simplicity
    description="Use this tool to run BLUP analysis on the dataset."
)

############################
# 5. Agent/Prompt Setup
############################
llm = OpenAI(temperature=0)  # more deterministic

prompt = PromptTemplate(
    template="""
You are an AI assistant specialized in statistical genetics using R.
You have access to the following tool:

Tool name: {tool_name}
Tool description: {tool_description}

The user will ask you questions or give instructions related to statistical genetics,
and you must decide if you should call the above tool to run a BLUP analysis.

You can output "Action: <tool_name>" followed by "Action Input: <input>"
if you wish to call the tool. Otherwise, provide a direct answer.

Begin interacting with the user:
{user_input}
""",
    input_variables=["tool_name", "tool_description", "user_input"]
)

def genetic_analysis_agent(user_input: str) -> str:
    agent_prompt = prompt.format(
        tool_name=blup_tool.name,
        tool_description=blup_tool.description,
        user_input=user_input
    )

    agent = ZeroShotAgent.from_llm_and_tools(
        llm=llm,
        tools=[blup_tool],
        verbose=True
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[blup_tool],
        verbose=True
    )
    response = agent_executor.run(agent_prompt)
    return response

############################
# 6. Execute Agent with a BLUP Query
############################
user_query = "Get BLUPs for grain yield across three environments using a mixed model."

result = genetic_analysis_agent(user_query)
print("\n===== Final Agent Response =====")
print(result)

