from streamlit.web import bootstrap

real_script = 'agents_setup.py'
bootstrap.run(real_script, f'run.py {real_script}', [], {})