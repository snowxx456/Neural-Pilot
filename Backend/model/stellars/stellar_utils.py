# model/stellar/stellar_utils.py
import os
import json
import hashlib
from stellar_sdk import Keypair, Server, TransactionBuilder, Network
from dotenv import load_dotenv

load_dotenv()

def register_pipeline_on_stellar(pipeline_id, dataset, model_name, accuracy, preprocessing_steps):
    secret = os.getenv("STELLAR_SECRET")
    keypair = Keypair.from_secret(secret)
    public_key = keypair.public_key

    server = Server("https://horizon-testnet.stellar.org")
    account = server.load_account(public_key)

    pipeline_data = {
        "dataset": dataset,
        "model": model_name,
        "accuracy": accuracy,
        "preprocessing": preprocessing_steps
    }
    pipeline_hash = hashlib.sha256(json.dumps(pipeline_data, sort_keys=True).encode()).hexdigest()

    memo_str = json.dumps({
        "m": model_name[:10],
        "a": round(accuracy, 2),
        "d": dataset[:9]
    })[:28]  # Stellar max memo length = 28 chars

    tx_builder = TransactionBuilder(
        source_account=account,
        network_passphrase=Network.TESTNET_NETWORK_PASSPHRASE,
        base_fee=100
    )

    tx_builder.add_text_memo(memo_str)
    tx_builder.append_manage_data_op(data_name=pipeline_id, data_value=pipeline_hash.encode())
    tx_builder.append_manage_data_op(data_name=f"{pipeline_id}_ds", data_value=dataset.encode())
    tx_builder.append_manage_data_op(data_name=f"{pipeline_id}_acc", data_value=str(accuracy).encode())
    tx_builder.append_manage_data_op(data_name=f"{pipeline_id}_model", data_value=model_name.encode())

    tx = tx_builder.set_timeout(30).build()
    tx.sign(keypair)
    response = server.submit_transaction(tx)

    return response["hash"]
