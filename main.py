from python_repo_template.function import random_sum

def main():
    print("Hello from python-repo-template!")
    n = int(input("Give me a number and I will add a random number to it:"))
    new_n = int(random_sum(n))
    print(f"Here is your number: {new_n}")


if __name__ == "__main__":
    main()


