from streamlit.web import bootstrap

real_script = 'app_url.py'
bootstrap.run(real_script, f'run.py {real_script}', [], {})