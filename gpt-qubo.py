import openai
import json
import numpy as np
openai.api_key = "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"


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
# def process_matrix(cost_matrix, cost_function, constraints):
def process_matrix(adj_matrix):
    """Get the current weather in a given location"""
    # print("flag")
    problem_info = {
        "adj_matrix": adj_matrix,
        # "cost_function": cost_function,
        # "constraints": constraints,
    }
    adj_matrix = np.array(json.loads(problem_info["adj_matrix"]))
    num_neurons = adj_matrix.shape[0]
    
    # print(np.array(json.loads(problem_info["adj_matrix"])))
    return json.dumps(problem_info)


def run_conversation():
    # Step 1: send the conversation and available functions to GPT
    messages = [{
                "role": "system", "content": "You are mathematician who extracts information from plain texts and reformulate the information into mathematical structures for further processing",
                "role": "user", "content": "You will be given a MaxCut problem based on plain texts. Construct an adjacency matrix and calculate the number of maxcuts based on the description of an undirected graph: 'Node 1 is connected to node 2 and node 3, node 2 is connected to node 3.'"}]
    functions = [
        {
            "name": "process_matrix",
            "description": "Calculate the maxcut for a given adjacency matrix",
            "parameters": {
                "type": "object",
                "properties": {
                    "adj_matrix": {
                        "type": "string",
                        "description": "The adjacency matrix to be further processed, e.g. M = [[0, 1], [0, 0]]"},
                        # "description": "The QUBO matrix that describes the energy hamiltonian of the specified problem, e.g. C = [[0, 10, 20, 5, 0], [10, 0, 5, 10, 0], [20, 5, 0, 2, 0], [5, 10, 2, 0, 0], [0, 0, 0, 0, 0]]"},
                    # "cost_function": {
                    #     "type": "string", 
                    #     "description": "The total energy of the problem that is to be optimized, e.g. SUM(i=1 to 5) SUM(j=1 to 5) C_ij * x_ij"},
                    # "constraints": {
                    #     "type": "string", 
                    #     "description": "The constraints that need to be satisfied on top of optimizing the energy function, e.g. SUM(j=1 to 5) x_ij = 1 for all i=1 to 5, SUM(i=1 to 5) x_ij = 1 for all j=1 to 5, u_i - u_j + 5 * x_ij <= 4 for all i=2 to 5 and j=2 to 5, x_ij in {0,1} for all i=1 to 5 and j=1 to 5, u_i is integer and 2 <= u_i <= 5 for all i=2 to 5, x_15 = 1 (Andreas starts at restaurant A on day 1), x_51 = 0 (No restaurant is visited after day 4)"}
                },
                "required": ["adj_matrix"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    # print(response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        print("called function")
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "process_matrix": process_matrix,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            adj_matrix=function_args.get("adj_matrix"),
            # cost_function=function_args.get("cost_function"),
            # constraints=function_args.get("constraints")
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response


run_conversation()