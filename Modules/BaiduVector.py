import requests
import json
import pprint


def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=tEUIklndR5n0EtmZBCTXGIMZ&client_secret=lUKitI49tkevMOiMl9NUIP6kpA7osNxN"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def main():
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + get_access_token()

    payload = json.dumps({
        "input": ["美食", "美食故事", "美食"]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(eval(response.text)['data'][0]['embedding'])
    print(eval(response.text)['data'][1]['embedding'])
    print(eval(response.text)['data'][2]['embedding'])


if __name__ == '__main__':
    main()