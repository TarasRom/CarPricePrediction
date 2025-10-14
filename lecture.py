import streamlit as st

st.title('Streamlit lecture notes')

st.header("Introduction")

st.markdown("""
In the rapidly evolving field of data science and analytics, the tools and platforms we use to explore data, build models,
and share insights are critical to our success and efficiency.

__Jupyter Notebooks__ have long been a staple in the data professional's toolkit, offering a convenient and interactive environment for code,visualization, and narrative.
However, as the demands for more dynamic, interactive, and user-friendly data applications grow, the limitations of Jupyter Notebooks in creating shareable,
production-ready applications become apparent.

Jupyter Notebooks are excellent for exploratory data analysis, prototyping, and educational purposes.
They provide a linear, cell-based interface where code, outputs, and markdown notes coexist, making them ideal for storytelling with data.
However, when it comes to turning these explorations into interactive web applications that can be easily shared with non-technical users, notebooks fall short.
They are not designed for creating standalone web applications, lack interactivity out of the box, and require substantial work to deploy in a user-friendly format.

This is where frameworks like `Streamlit`, `Dash`, and `Panel` come into play, offering powerful alternatives for building and deploying data applications. They empower users to transform complex datasets into user—friendly web applications, whether mere numbers or intricate charts.

The beauty of these technologies is that they not only allow experts to analyze and visualize data but enable anyone, regardless of their technical expertise,
to interact with and grasp the insights presented. So, if you want to showcase your data engaging and comprehensibly, these tools offer the perfect palette.
""")


st.header("Install & run Streamlit")
st.write("It doesn't take long to start developing with Streamlit, since you don't even need any front-end web development experience.\
         With Streamlit, you script everything with Python. Streamlit is also compatible with data science libraries that you probably know.\
         \nWith your Python environment ready, Streamlit installation is simple:")
st.code("pip install streamlit", language="bash")

st.markdown("Assuming that you have written Streamlit code in a file called `app.py`, you can run the application with:")
st.code("streamlit run app.py", language="python")
st.markdown("`run app.py`  will start running the application on your local machine, and provide a link that you can use to access the app on the network.")

st.divider()

st.header("Streamlit Widgets")
st.subheader("Displaying text")
st.markdown("""
Streamlit has several widgets for displaying text, for example:

* `st.text` displays fixed-width and preformatted text
* `st.markdown` shows markdown text
* `st.latex` displays mathematical expressions formatted as LaTex
* `st.write` behaves differently depending on the inputs given to it. For example:
    * when you pass a Pandas DataFrame to it, it prints the data frame as a table
    * displays information about a function when a function is passed to it
    * displays a Keras model when one is passed to it
* `st.title` displays text in title formatting
* `st.header` displays text in header formatting
* `st.code` displays code
* `with st.echo()` displays the code after it as well as runs it.

> Note: most of the time you can use `write` instead of any other displaying function for text and data.

You have already seen most of the text widgets in action, but here's a more condensed example and code to copy:
""")
with st.echo():
    st.code("st.text()", language='python')
    st.text('Formatted text')
    st.code("st.markdown()", language='python')
    st.markdown('# This is Heading 1 in Markdown')
    st.code("st.title()", language='python')
    st.title('This is a title')
    st.code("st.header()", language='python')
    st.header('Header')
    st.code("st.subheader()", language='python')
    st.subheader('Sub Header')
    st.code("st.latex()", language='python')
    st.latex(r'''
            a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
            sum_{k=0}^{n-1} ar^k =
            a \left(\frac{1-r^{n}}{1-r}\right)
            ''')
    st.markdown("> Note that raw string `r` prefix to a string is necessary for LaTex as it uses special symbols")
    st.code("st.write()", language='python')
    st.write('Can display many things')


st.subheader("User interaction")
st.markdown("""
Streamlit also has widgets that let users interact with your application, for example:

* you can use the select box to let the user choose between several options (say, enable the user to filter data depending on a certain category)
* the multi-select widget is similar to the select box, but allows multiple selections
* the text area and text input widgets can be used to collect user input
* the date and time input can be used to collect time and date input
* you can also let users upload files using the file uploader widget
 (this can come in handy when you've built an image classifier or object detection model, and want users to upload images and see the result)
""")

with st.echo():
    st.button('Click here')
    st.checkbox('Check')
    st.radio('Radio', [1,2,3])
    st.selectbox('Select', [1,2,3])
    st.multiselect('Multiple selection', [21,85,53])
    st.slider('Slide', min_value=10, max_value=20)
    st.select_slider('Slide to select', options=[1,2,3,4])
    st.text_input('Enter some text')
    st.number_input('Enter a number')
    st.text_area('Text area')
    st.date_input('Date input')
    st.time_input('Time input')
    st.file_uploader('File uploader')
    st.color_picker('Color Picker')

st.write("The returned value from these widgets represents the selected option or uploaded file bytes")
with st.echo():
    x = st.slider('This value is squared', min_value=10, max_value=20)
    st.write(x, 'squared is', x * x)

st.divider()

st.subheader("Displaying progress")
st.markdown("""
When building your application, it's always good practice to show user progress or some status.
For example, when loading a large dataset, you can show a progress bar.

Some other status and progress widgets that you can use in Streamlit include:

* `st.spinner()` displays a temporary message when executing a block of code
* `st.balloons()` shows celebratory balloons
* `st.error()` displays an error message
* `st.warning` shows a warning message
* `st.info` shows informational messages
* `st.success` shows success messages 
* `st.exception` communicates an exception in your application
""")
with st.echo():
    import time
    if st.button('Click here to display progress'):
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        st.spinner()
        with st.spinner(text='In progress'):
            time.sleep(5)
            st.success('Done')
        st.balloons()
        st.error('Error message')
        st.warning('Warning message')
        st.info('Info message')
        st.success('Success message')
        e = RuntimeError('This is an exception of type RuntimeError')
        st.exception(e)
    else:
        st.write("You have not clicked the button yet.")

st.divider()


st.header("Data visualization")

st.write("Streamlit supports many visualization libraries and their features, as well as has some built-in visualization functions. \
         Most of them can be found in documentation, just some common examples:")

tab_core, tab_mpl, tab_sns, tab_plotly = st.tabs(["Streamlit", "Matplotlib", "Seaborn", "Plotly"])

with tab_core:
    with st.echo():
        import pandas as pd
        import numpy as np

        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

        st.line_chart(chart_data)

with tab_mpl:
    with st.echo():
        import matplotlib.pyplot as plt
        import numpy as np
        days = np.arange(1,31)
        candidate_A = 50+days*0.07+2*np.random.randn(30)
        candidate_B = 50-days*0.1+3*np.random.randn(30)

        ymin = min(candidate_A.min(),candidate_B.min())
        ymax = max(candidate_A.max(),candidate_B.max())
        # Set style
        plt.style.use('fivethirtyeight')

        fig = plt.figure(figsize=(12,5))
        plt.title("Time series plot of poll percentage over a month\n",
                  fontsize=20, fontstyle='italic')
        plt.xlabel("Days",fontsize=16)
        plt.ylabel("Poll percentage (%)",fontsize=16)
        plt.grid (True)
        plt.ylim(ymin*0.98,ymax*1.02)
        plt.xticks([i*2 for i in range(16)],fontsize=14)
        plt.yticks(fontsize=15)

        # Main plotting function - plot (note markersize, lw (linewidth) arguments)
        plt.plot(days,candidate_A,'o-',markersize=10,c='blue',lw=2)
        plt.plot(days,candidate_B,'^-',markersize=10,c='green',lw=2)

        plt.legend(['Poll percentage of candidate A (%)',
                    'Poll percentage of candidate B (%)'],loc=2,fontsize=14)

        # Pass the figure to Streamlit
        st.pyplot(fig)

with tab_sns:
    with st.echo():
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")

        fig = plt.figure()

        rs = np.random.RandomState(365)
        values = rs.randn(365, 4).cumsum(axis=0)
        dates = pd.date_range("1 1 2024", periods=365, freq="D")
        data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
        data = data.rolling(7).mean()

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        # Pass the figure to Streamlit
        st.pyplot(fig)


st.divider()

st.header("Summary")
st.write("""
We delved into Streamlit fundamental capabilities, focusing on how this innovative tool can transform your data scripts into interactive, web-based applications with minimal effort.
We covered the essence of Streamlit's appeal: its intuitive use of widgets for interactivity, its efficient data presentation techniques, and its powerful visualization options.
These features enable even those with limited web development experience to quickly create and share dynamic data applications.

However, our journey into Streamlit has merely scratched the surface. Beyond these basic concepts lies a wealth of advanced features designed to enhance your applications further,
streamline your development process, and create more sophisticated and user-friendly data experiences.

__Advanced Features Awaiting Discovery:__

- __State Management:__ Learn how to manage session states within your application, enabling more complex interactions and data persistence across user sessions.
- __Custom Themes and Layouts:__ Dive deeper into customizing the look and feel of your applications, tailoring the user interface to meet specific needs or align with brand guidelines.
- __Security and Authentication:__ Explore options for adding authentication layers to your Streamlit applications, ensuring that sensitive data remains protected and accessible only to authorized users.
- __Deployment Strategies:__ Gain insights into various deployment options and best practices for making your Streamlit applications available and scalable across different environments.
- __Integration with Other Services:__ Discover how to integrate your Streamlit applications with external APIs, databases, and services, expanding the functionality and interactivity of your tools.

In summary, while Jupyter Notebooks remain invaluable for data exploration and documentation, the advent of Streamlit, Dash, and Panel opens up new possibilities for bringing data science projects to life.
These frameworks empower data professionals to create interactive, engaging applications that communicate insights more effectively and reach a wider audience.
""")