# import curses
# import datetime
#
# stdscr = curses.initscr()
# curses.noecho()
# stdscr.nodelay(1) # set getch() non-blocking
#
# stdscr.addstr(0,0,"Press \"p\" to show count, \"q\" to exit...")
# line = 1
# try:
#     while 1:
#         c = stdscr.getch()
#         if c == ord('p'):
#             stdscr.addstr(line,0,"Some text here")
#             line += 1
#         elif c == ord('q'): break
#
#         """
#         Do more things
#         """
#
# finally:
#     curses.endwin()
import numpy as np
def nonlin(x, deriv=False):
    if deriv:
        return (x*(1-x))
    return 1/(1+np.exp(-x))
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
np.random.seed(1)
nums = [2, 4, 1]
network = [2*np.random.random((nums[i]+1,nums[i+1]))-1 for i in range(len(nums)-1)]
print('network', network)
for j in range(100000):
    outputs = [X]
    for layer in network:
        outputs[-1] = np.c_[outputs[-1], np.ones(len(outputs[-1]))]
        outputs.append(nonlin(np.dot(outputs[-1], layer)))
    print('outputs', outputs, '\n')
    errors = [y - outputs[2]]
    print('errors', errors)
    # if(j % 100000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
    #     print('outputs, prediction', l0, l1, l2, y, l2.shape)
    #     print('weights',  self.network0, self.network1)
    #     print("Error: " + str(np.mean(np.abs(errors[2]))))
        # print('Training input l0:', l0, '\nDot product between training and rand:', np.dot(l0, self.network0), 'non linear dot product l1:', l1, '\n dot product between l1, and self.network1:', np.dot(l1, self.network1), 'nonlinear dot product between l1, and self.network1:', l2, 'input and output training data: ', self.network0, self.network1, errors[2], nonlin(l2, deriv=True))
    deltas = [errors[-1]*nonlin(outputs[2], deriv=True)]
    print('deltas', deltas)
    # if(j % 100000) == 0:
    #     print('l2Error, nonlin\'(l2)', errors[2], nonlin(l2, deriv=True))
    #     print('l2Delta, self.network1.t', l2_delta, self.network1.T)
    for i in range(len(network)-1):
        errors.insert(0, deltas[0].dot(network[i+1].T))
        print('layer', i, 'error', errors[0])
        # if(j % 100000) == 0:
        #     print('l1Error', errors[1])
        # print(nonlin(outputs[i+1],deriv=True))
        deltas.insert(0, errors[0] * nonlin(outputs[i+1],deriv=True))
        print('layer', i, 'delta', deltas[0], '\n')
        # if(j % 100000) == 0:
        #     print('self.network1, l1.T, l2Delta', network[1].shape, outputs[1].T.shape, deltas[1].shape)
        # if(j % 100000) == 0:
        #     print('self.network0, l0.T, l1Delta', network[0].shape, outputs[0].T.shape, deltas[0].shape)
    #update weights (no learning rate term)
    for i in range(len(deltas)):
        delta = outputs[i].T.dot(deltas[i])
        print(delta,'\n', network[i])
        network[i] += delta

print("Output after training")
print(outputs[2])
