import streamlit as st
import os
from PIL import Image
from backend1 import chatbot_with_plots, chatbot  # Fixed import - removed plot_saving_python_repl
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': '1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

def display_message(message):
    """Helper function to display a message with plots if available"""
    # Display text content
    if message.get('content'):
        st.text(message['content'])
    
    # Display plots if available
    if message.get('plot_paths'):
        for plot_path in message['plot_paths']:
            if os.path.exists(plot_path):
                try:
                    image = Image.open(plot_path)
                    st.image(image, caption="Generated Plot", use_column_width=True)
                except Exception as e:
                    st.error(f"Could not display plot: {e}")
            else:
                st.warning(f"Plot file not found: {plot_path}")

# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        display_message(message)

user_input = st.chat_input('Ask me to create a visualization from your database!')

# Handle example query selection
if 'example_query' in st.session_state:
    user_input = st.session_state['example_query']
    del st.session_state['example_query']

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({
        'role': 'user', 
        'content': user_input
    })
    
    with st.chat_message('user'):
        st.text(user_input)

    # Show loading spinner while processing
    with st.spinner('Creating your visualization...'):
        try:
            # Get response with plots
            result = chatbot_with_plots(
                {'messages': [HumanMessage(content=user_input)]}, 
                config=CONFIG
            )
            
            response = result['response']
            plot_paths = result['plot_paths']
            
            ai_message = response['messages'][-1].content
            
            # Show debug info if enabled
            debug_mode = st.session_state.get('debug_mode', False)
            if debug_mode:
                with st.expander("🐛 Debug Information"):
                    st.write("**Plot paths returned:**", plot_paths)
                    st.write("**Response structure:**", type(response))
                    if plot_paths:
                        for path in plot_paths:
                            st.write(f"• File exists: {os.path.exists(path)} - {path}")
                            if os.path.exists(path):
                                st.write(f"  File size: {os.path.getsize(path)} bytes")
            
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            debug_mode = st.session_state.get('debug_mode', False)
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
            ai_message = f"Sorry, I encountered an error: {str(e)}"
            plot_paths = []

    # Display assistant response
    with st.chat_message('assistant'):
        st.text(ai_message)
        
        # Display any plots that were created
        if plot_paths:
            st.success(f"✅ Created {len(plot_paths)} visualization(s):")
            for i, plot_path in enumerate(plot_paths):
                if os.path.exists(plot_path):
                    try:
                        image = Image.open(plot_path)
                        st.image(image, caption=f"Visualization {i+1}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not display plot {i+1}: {e}")
                else:
                    st.warning(f"Plot file not found: {plot_path}")
        else:
            if "Generated 0 plot(s)" in ai_message:
                st.info("🔄 No plots were generated. Try a different visualization request.")
                
                # Show some helpful suggestions
                with st.expander("💡 Troubleshooting Tips"):
                    st.write("Try these types of requests:")
                    st.write("• 'Create a scatter plot of temperature vs salinity'")
                    st.write("• 'Show me a histogram of ocean depths'")
                    st.write("• 'Make a bar chart of average pH by region'")
                    st.write("• 'Plot temperature distribution'")
    
    # Add assistant message to history with plot paths
    st.session_state['message_history'].append({
        'role': 'assistant', 
        'content': ai_message,
        'plot_paths': plot_paths if plot_paths else []
    })

# Add some example prompts and debug options
with st.sidebar:
    st.header("🎨 Visualization Assistant")
    st.write("Create data visualizations from your oceanographic database!")
    
    # Debug mode toggle
    debug_mode = st.checkbox("🐛 Debug Mode", help="Show detailed debugging information")
    st.session_state['debug_mode'] = debug_mode
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state['message_history'] = []
        st.rerun()
    
    st.header("📊 Example Queries")
    example_queries = [
        "Create a scatter plot of temperature vs salinity",
        "Show me a histogram of ocean depths", 
        "Make a bar plot of pH levels by region",
        "Plot temperature distribution",
        "Create a scatter plot of latitude vs longitude"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
            st.session_state['example_query'] = query
            st.rerun()
    
    st.header("📋 Database Info")
    st.write("**Table:** argo_profiles")
    st.write("**Columns:**")
    st.write("• float_id, date")
    st.write("• latitude, longitude, depth")  
    st.write("• temperature, salinity, pressure")
    st.write("• oxygen, pH, conductivity")
    st.write("• region")
    
    # Show plot directory info
    plots_dir = "plots"
    if os.path.exists(plots_dir):
        png_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        st.write(f"**Plots created:** {len(png_files)}")
        if debug_mode and png_files:
            st.write("Recent plot files:")
            for f in sorted(png_files)[-5:]:  # Show last 5
                st.write(f"• {f}")