
import random
import math


class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.fitness = 0

    def is_terminal(self):
        return self.left is None and self.right is None

    def evaluate(self, x):
        if self.is_terminal():
            if self.value == "x":
                return x if not isinstance(x, dict) else x["x"]

            if self.value in ("x1", "x2", "x3"):
                return x[self.value]

            return float(self.value)

        left_value = self.left.evaluate(x)
        right_value = self.right.evaluate(x)

        if self.value == "+":
            result = left_value + right_value
        elif self.value == "-":
            result = left_value - right_value
        elif self.value == "*":
            result = left_value * right_value
        elif self.value == "/":
            # Protected division
            if abs(right_value) < 1e-10:
                result = left_value
            else:
                result = left_value / right_value
        else:
            raise ValueError(f"Unknown operator: {self.value}")

        if not math.isfinite(result):
            raise ValueError("Non-finite result")

        return result

    def depth(self):
        if self.is_terminal():
            return 1
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self):
        if self.is_terminal():
            return 1
        return 1 + self.left.size() + self.right.size()

    def copy(self):
        if self.is_terminal():
            new_node = Node(self.value)
        else:
            new_node = Node(
                self.value,
                self.left.copy(),
                self.right.copy()
            )

        new_node.fitness = self.fitness
        return new_node

    # Print tree
    def __str__(self):
        return "\n".join(line.rstrip() for line in self._render()[0])

    def _render(self):
        label = str(self.value)

        if self.is_terminal():
            return [label], len(label), (len(label) - 1) // 2

        l_lines, l_w, l_c = self.left._render()
        r_lines, r_w, r_c = self.right._render()

        gap = 3
        if (l_w + gap + r_c - l_c) % 2:
            gap += 1

        r_c += l_w + gap
        mid = (l_c + r_c) // 2
        k = (mid - l_c + 1) // 2

        height = max(len(l_lines), len(r_lines))
        l_lines = l_lines + [" " * l_w] * (height - len(l_lines))
        r_lines = r_lines + [" " * r_w] * (height - len(r_lines))

        body = [a + " " * gap + b for a, b in zip(l_lines, r_lines)]
        width = l_w + gap + r_w

        start = mid - (len(label) - 1) // 2
        pad_l = max(0, -start)
        pad_r = max(0, start + len(label) - width)

        start, mid, width = (
            start + pad_l,
            mid + pad_l,
            width + pad_l + pad_r
        )

        body = [" " * pad_l + line + " " * pad_r for line in body]

        top = (" " * start + label).ljust(width)
        branches = (
            " " * (mid - k) + "/" +
            " " * (2 * k - 1) + "\\"
        ).ljust(width)

        return [top, branches] + body, width, mid

    def to_infix(self):
        if self.is_terminal():
            return str(self.value)

        return (
            f"({self.left.to_infix()} "
            f"{self.value} "
            f"{self.right.to_infix()})"
        )


def get_all_nodes(node, nodes):
    nodes.append(node)

    if not node.is_terminal():
        get_all_nodes(node.left, nodes)
        get_all_nodes(node.right, nodes)

    return nodes


def get_all_inner_nodes(node, nodes):
    if node.is_terminal():
        return nodes

    nodes.append(node)
    get_all_inner_nodes(node.left, nodes)
    get_all_inner_nodes(node.right, nodes)

    return nodes


def generate_random_tree(
    operators,
    depth=5,
    p_terminal=0.2,
    p_x=0.5,
    variables=None,
    real_constants=False
):
    if variables is None:
        variables = ["x"]

    # Depth applies only to initial generation,
    # not to crossover or mutation.
    if depth <= 1 or random.random() < p_terminal:
        if random.random() < p_x:
            return Node(random.choice(variables))

        if real_constants:
            return Node(str(round(random.uniform(-10, 10), 2)))

        return Node(str(random.randint(-10, 10)))

    return Node(
        random.choice(operators),
        generate_random_tree(
            operators, depth - 1, p_terminal,
            p_x, variables, real_constants
        ),
        generate_random_tree(
            operators, depth - 1, p_terminal,
            p_x, variables, real_constants
        )
    )


def mutate(
    node,
    operators,
    p_x=0.5,
    variables=None,
    real_constants=False
):
    if variables is None:
        variables = ["x"]

    # Work on a copy so the parent is unchanged.
    child = node.copy()

    nodes = get_all_nodes(child, [])
    mutate_node = random.choice(nodes)

    # Occasionally replace an entire subtree.
    # There is no restriction on the resulting tree size.
    if random.random() < 0.15:
        new_subtree = generate_random_tree(
            operators,
            depth=random.randint(2, 5),
            p_terminal=0.3,
            p_x=p_x,
            variables=variables,
            real_constants=real_constants
        )

        mutate_node.value = new_subtree.value
        mutate_node.left = new_subtree.left
        mutate_node.right = new_subtree.right

    elif mutate_node.is_terminal():

        # If a numeric constant is selected, sometimes
        # adjust it slightly instead of replacing it.
        if real_constants and mutate_node.value not in variables:
            if random.random() < 0.5:
                old_value = float(mutate_node.value)
                new_value = old_value + random.gauss(0, 0.5)
                mutate_node.value = str(round(new_value, 4))
            else:
                if random.random() < p_x:
                    mutate_node.value = random.choice(variables)
                else:
                    mutate_node.value = str(
                        round(random.uniform(-10, 10), 2)
                    )

        else:
            if random.random() < p_x:
                mutate_node.value = random.choice(variables)
            elif real_constants:
                mutate_node.value = str(
                    round(random.uniform(-10, 10), 2)
                )
            else:
                mutate_node.value = str(
                    random.randint(-10, 10)
                )

    else:
        # Change an operator.
        mutate_node.value = random.choice(operators)

    child.fitness = 0
    return child


def crossover(parent1, parent2):
    child = parent1.copy()
    donor = parent2.copy()

    donor_nodes = get_all_nodes(donor, [])
    donor_subtree = random.choice(donor_nodes).copy()

    child_nodes = get_all_nodes(child, [])
    target = random.choice(child_nodes)

    target.value = donor_subtree.value
    target.left = donor_subtree.left
    target.right = donor_subtree.right

    # No depth or size checks.
    child.fitness = 0
    return child


if __name__ == "__main__":
    operators = ["+", "-", "*", "/"]

    parent1 = generate_random_tree(operators)
    parent2 = generate_random_tree(operators)

    print("Parent 1:", parent1.to_infix())
    print("Parent 2:", parent2.to_infix())

    child = crossover(parent1, parent2)
    print("Crossover:", child.to_infix())

    mutated = mutate(child, operators)
    print("Mutation:", mutated.to_infix())

    print("Depth:", mutated.depth())
    print("Size:", mutated.size())
    print("Evaluation:", mutated.evaluate(2))

