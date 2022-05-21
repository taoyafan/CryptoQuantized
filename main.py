from api_key import API_KEY, SECRET_KEY
from binance.client import Client

def main(client: Client):
    # get market depth
    depth = client.get_order_book(symbol='BNBBTC')
    print(depth)

if __name__ == "__main__":
    
    # Note: Need to enable http proxy in v2rayN
    proxies = {
        "http": "http://127.0.0.1:8900",
        "https": "http://127.0.0.1:8900",
    }
    # client = Client(API_KEY, SECRET_KEY, {'proxies': proxies})
    client = Client(API_KEY, SECRET_KEY, testnet=True)
    main(client)
    print("Finished")