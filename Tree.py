import csv
import random
import math


class Node:
  
    def __init__(self, value, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right

  //only one

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
          //incase of zero

          if abs(right_value) < 1e  - 10:
            return left_value

          else return left_value / right_value

//find depth

    def depth(self):

      if self.is_terminal():
        return 1

      return 1 + max(self.left.depth(), self.right.depth())

//size

    def size(self):

      if self.is_terminal():
        return 1

      return 1 + self.left.size() + self.right.size()

 

  //copy of tree

    def copy(self):

      if self.is_terminal():
        return Node(self.value)

      return Node (self.value,
                   self.left.copy(),
                   self.right.copy())
      
                   
            

       

   
