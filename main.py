import sys
import os

from llm.generator import get_generator

def main():
    print("Initializing Generator...")
    generator = get_generator()
    
    question = "What is the capital of France?"
    print(f"\nQuestion: {question}")
    print("Generating response...")
    
    output = generator.generate(question, max_new_tokens=20)
    
    print(f"\nAnswer: {output.answer}")

if __name__ == "__main__":
    main()
