
def build_expression_graph(leaves: list, operations: list) -> tuple:
    nodes = []
    values = {}

    # 1. Create leaf nodes
    for leaf in leaves:
        node = {
            "id": leaf["id"],
            "data": float(leaf["data"]),
            "grad": 0.0,
            "op": "",
            "parents": []
        }

        nodes.append(node)
        values[node["id"]] = node["data"]

    # 2. Create operation nodes
    for operation in operations:
        node_id = operation["id"]
        op = operation["op"]
        left = operation["left"]
        right = operation["right"]

        x = values[left]
        y = values[right]

        # Compute the forward value
        if op == "+":
            result = x + y
        else:  # op == "*"
            result = x * y

        node = {
            "id": node_id,
            "data": float(result),
            "grad": 0.0,
            "op": op,
            "parents": [left, right]
        }

        nodes.append(node)
        values[node_id] = node["data"]

    # 3. Return nodes and final node ID
    return nodes, nodes[-1]["id"]
