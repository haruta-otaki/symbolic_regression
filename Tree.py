import csv
import random
import math

class Node:
  
    def __init__(self, value, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right

    def is_terminal(self):
        return self.left is None and self.right is None

    def evaluate(self, x):
       

    
        if self.value == "x":
            return x

        
        if self.is_terminal():
            return float(self.value)

        
        left_value = self.left.evaluate(x)
        right_value = self.right.evaluate(x)

        if self.value == "+":
            return left_value + right_value

        elif self.value == "-":
            return left_value - right_value

        elif self.value == "*":
            return left_value * right_value

       

   
