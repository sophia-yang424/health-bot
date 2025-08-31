## Chatbot Example

### To run the chatbot:
1. Create a new conda environment using this command: `conda env create -f environment.yml`
2. copy .env.example file to a new file named .env, then fill in all the required values for environment variables in the new .env file.
3. Install PostgreSQL Database on your local computer. PostgreSQL download URL: https://www.postgresql.org/download/
4. Install DBeaver Community Edition (NOT the PRO edition) on your local computer. DBeaver download URL: https://dbeaver.io/download/
5. Set up the DBeaver app to connect to your PostgreSQL databae
6. In DBeaver, execute the SQL statements in create_chainlit_data_layer_tables.sql file to create the tables needed for Chainlit data layer for data persistence
7. Uncomment line 196 in Chatbot.py for the first time run. Line 196 is `# await checkpointer.setup()`. line 196 need to run ONLY ONCE. So comment it out after the first time run.
8. Run `chainlit run chatbot_example.py` to start the chatbot. It should automatically open the Chatbot login page in your default web browser. Use "admin" for your user name (no password needed) - for development purpose only, NOT FOR PRODUCTION USE. Secure authentication and authorization methods are necessary for production deployment.