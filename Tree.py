import csv
import random
import math

class Node:
    def __init__(self, operators, max_depth = 4):
        self.value = None
        self.left = None
        self.right = None
        self.operators = operators
        depths = list(range(1, max_depth + 1))
        self.depth = random.choice(depths)
        
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
        elif self.value == "/":
            #incase of zero
            if abs(right_value) < 1e  - 10:
                return left_value
            else:
                return left_value / right_value
        else:
            raise ValueError(f"Unknown operator: {self.value}")

#find depth
    def depth(self):
      if self.is_terminal():
        return 1
      return 1 + max(self.left.depth(), self.right.depth())

#size
    def size(self):
      if self.is_terminal():
        return 1
      return 1 + self.left.size() + self.right.size()

 

  #copy of tree
    def copy(self):
      if self.is_terminal():
        return Node(self.value)
      return Node (self.value,
                   self.left.copy(),
                   self.right.copy())

#print tree
    def __str__(self):
      if self.is_terminal():
        return str(self.value)
      return f"({self.left} {self.value} {self.right})"
            
    def generate_random_tree(self, node, depth):
        if depth == 0:
            p = random.random()
            if p < 0.5: 
                node = Node("x")
            else:
                node = Node(str(random.uniform(-10, 10)))
            return
        else: 
            node = Node(random.choice(self.operators))
            self.generate_random_tree(node.left, depth - 1)
            self.generate_random_tree(node.right, depth - 1)
        return
