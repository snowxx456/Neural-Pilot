from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
groq_api_key = user_secrets.get_secret("Groq")