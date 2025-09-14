import os
import getpass
import uuid
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage, AIMessage
import re
import glob
import time

load_dotenv()

# Create plots directory if it doesn't exist
PLOTS_DIR = "plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Database setup
db = SQLDatabase.from_uri("sqlite:///argo.db")

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain.chat_models import init_chat_model
llm = init_chat_model("gemma2-9b-it", model_provider="groq")

# Much simpler approach - just count files before and after
def execute_with_plot_detection(code):
    """Execute code and detect any new plot files created"""
    
    # Get existing files before execution
    existing_files = set(glob.glob(os.path.join(PLOTS_DIR, "*.png")))
    
    # Execute the code
    python_repl = PythonREPL()
    
    # Generate a unique filename
    timestamp = str(int(time.time()))
    plot_filename = os.path.join(PLOTS_DIR, f"plot_{timestamp}.png")
    
    # Force save the plot with our filename
    modified_code = code + f"""
plt.savefig('{plot_filename}', dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_filename}")
"""
    
    print(f"Executing code:\n{modified_code}")
    result = python_repl.run(modified_code)
    print(f"Execution result: {result}")
    
    # Check if our file was created
    if os.path.exists(plot_filename):
        print(f"SUCCESS: Plot file created at {plot_filename}")
        return [plot_filename], result
    else:
        print(f"WARNING: Expected plot file not found at {plot_filename}")
        # Fallback: check for any new files
        current_files = set(glob.glob(os.path.join(PLOTS_DIR, "*.png")))
        new_files = current_files - existing_files
        new_files_list = list(new_files)
        print(f"New files detected: {new_files_list}")
        return new_files_list, result

# Custom visualization agent
class VisualizationAgent:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        
    def create_visualization(self, user_request: str):
        """Main method to create visualization based on user request"""
        try:
            # Step 1: Get data from database
            print("Step 1: Querying database...")
            sql_query = self._generate_sql_query(user_request)
            print(f"Generated SQL: {sql_query}")
            
            data = self.db.run(sql_query)
            print(f"Retrieved data: {data}")
            
            # Step 2: Generate Python visualization code
            print("Step 2: Generating visualization code...")
            python_code = self._generate_visualization_code(user_request, data)
            print(f"Generated code: {python_code}")
            
            # Step 3: Execute the code and detect plots
            print("Step 3: Executing visualization code...")
            plot_files, exec_result = execute_with_plot_detection(python_code)
            print(f"Plot files created: {plot_files}")
            
            return {
                'success': True,
                'message': f"Created visualization successfully! Generated {len(plot_files)} plot(s).",
                'plots': plot_files,
                'data': data,
                'sql': sql_query,
                'code': python_code,
                'exec_result': exec_result
            }
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f"Error creating visualization: {str(e)}",
                'plots': [],
                'error': str(e)
            }
    
    def _generate_sql_query(self, user_request: str) -> str:
        """Generate SQL query based on user request"""
        prompt = f"""
        Generate a SQL query for the table 'argo_profiles' based on this request: "{user_request}"
        
        Available columns: float_id, date, latitude, longitude, depth, temperature, salinity, pressure, oxygen, pH, conductivity, region
        
        Rules:
        - LIMIT results to 5 rows for performance
        - Select only the columns needed for the visualization
        - Return ONLY the SQL query, nothing else
        
        Examples:
        - "temperature vs salinity" -> "SELECT temperature, salinity FROM argo_profiles LIMIT 5"
        - "depth histogram" -> "SELECT depth FROM argo_profiles LIMIT 5"
        - "pH by region" -> "SELECT pH, region FROM argo_profiles LIMIT 5"
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql = response.content.strip()
        
        # Clean up the response
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = sql.strip()
        
        return sql
    
    def _generate_visualization_code(self, user_request: str, data: str) -> str:
        """Generate Python visualization code"""
        prompt = f"""
        Generate Python matplotlib code to visualize this data: {data}
        User request: "{user_request}"
        
        IMPORTANT RULES:
        1. Start with these exact imports:
           import matplotlib.pyplot as plt
           import ast
        2. Parse data using: data = ast.literal_eval('''{data}''')
        3. Create the plot based on the user request
        4. Set figure size: plt.figure(figsize=(10, 6))
        5. Add proper labels and title
        6. End with plt.show()
        7. Return ONLY Python code, no explanations, no markdown blocks
        
        Example for scatter plot:
        import matplotlib.pyplot as plt
        import ast
        data = ast.literal_eval('''{data}''')
        plt.figure(figsize=(10, 6))
        x_vals = [item[0] for item in data]
        y_vals = [item[1] for item in data]
        plt.scatter(x_vals, y_vals)
        plt.xlabel('Column 1')
        plt.ylabel('Column 2')
        plt.title('Scatter Plot')
        plt.show()
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        code = response.content.strip()
        
        # Clean up any markdown
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        code = code.strip()
        
        return code

# Initialize the agent
viz_agent = VisualizationAgent(db, llm)

# Wrapper function for Streamlit
def chatbot_with_plots(messages, config=None):
    user_message = messages['messages'][0].content
    result = viz_agent.create_visualization(user_message)
    
    print(f"Final result: {result}")
    
    if result['success']:
        response_content = result['message']
    else:
        response_content = f"Error: {result['message']}"
    
    # Create a mock response similar to what LangGraph would return
    mock_response = {
        'messages': [
            HumanMessage(content=user_message),
            AIMessage(content=response_content)
        ]
    }
    
    return {
        'response': mock_response,
        'plot_paths': result['plots']
    }

# Backward compatibility
chatbot = viz_agent

print("Custom visualization agent setup complete!")