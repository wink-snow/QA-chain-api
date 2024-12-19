import json

class KeyChecker:
    def __init__(self, id):
       self.id = id
       self.users = self.get_keys()

    def get_keys(self):
        with open('app\.key', 'r') as f:
            data = json.load(f)
        return data["users"]
    
    def check(self) -> int:
        for user in self.users:
            if user["id"] == self.id:
                return self.users.index(user)
        return -1

if __name__ == "__main__":
    key_checker = KeyChecker("123456")
    if key_checker.check() != -1:
        print("key is valid")
    else:
        print("key is invalid")