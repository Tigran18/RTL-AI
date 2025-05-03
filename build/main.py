from ai_module import Network, JSON

net = Network([3, 4, 1], 0.1, 1000)
net.forward_propagate([1.0, 0.5, -0.3])
net.display_outputs()


json_obj = JSON('{"x": 42}', True)
json_obj.print(0)
