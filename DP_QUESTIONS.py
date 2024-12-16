# Fibonacci number
# Find nth fibonacci number

# n = 8
# def solve_rec(n):
#     if n == 0 or  n == 1:
#         return n
#     return solve_rec(n - 1) + solve_rec(n - 2)

# def solve_mem(n, dp):
#     if n == 0 or  n == 1:
#         return n
#     if dp[n] != -1:
#         return dp[n]
#     dp[n]  = solve_mem(n - 1, dp) + solve_mem(n - 2, dp)
#     return dp[n]

# def solve_tab(n):
#     dp = [-1] * (n + 1)
#     dp[0] = 0
#     dp[1] = 1
#     for i in range(2, n + 1):
#         dp[i] = dp[i - 1] + dp[i - 2]
#     return dp[n]

# def solve_SO(n):
#     # dp = [-1] * (n + 1)
#     prev1 = 1
#     prev2 = 0
#     for _ in range(2, n + 1):
#         curr = prev1 + prev2
#         prev2 = prev1
#         prev1 = curr
#     return curr

# def nth_fibonacci_number(n):
#     # ans = solve_rec(n)
#     # dp = [-1] * (n + 1) 
#     # ans = solve_mem(n, dp)
#     # ans = solve_tab(n)
#     ans = solve_SO(n)
#     return ans

# print(nth_fibonacci_number(n))

# # climbing stairs
# # You are climbing a staircase. It takes n steps to reach the top.

# # Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

# n = 8

# def solve_rec(n):
#     if n < 0:
#         return 0
#     if n == 0:
#         return 1
#     return solve_rec(n - 1) + solve_rec(n - 2)

# def solve_mem(n, dp):
#     if n < 0:
#         return 0
#     if n == 0:
#         return 1
#     if dp[n] != -1:
#         return dp[n]
#     dp[n] = solve_mem(n - 1, dp) + solve_mem(n - 2, dp)
#     return dp[n]

# def solve_tab(n):
#     dp = [-1 ] * (n + 1)
#     dp[0] = 1
#     dp[1] = 1
#     for i in range(2, n + 1):
#         dp[i] = dp[i - 1] + dp[i - 2]
#     return dp[n]

# def solve_SO(n):
#     prev1 = 1
#     prev2 = 1
#     for i in range(2, n + 1):
#         curr = prev1 + prev2
#         prev2 = prev1
#         prev1 = curr
#     return prev1


# def climb_stairs(n):
#     ans = solve_rec(n)
#     # dp = [-1] * (n + 1)
#     # ans = solve_mem(n, dp)
#     # ans = solve_tab(n)
#     ans =  solve_SO(n)
#     return ans

# print(climb_stairs(n))


# # Minimum cost climbing stairs
# # You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

# # You can either start from the step with index 0, or the step with index 1.

# # Return the minimum cost to reach the top of the floor.

# cost = [1,100,1,1,1,100,1,1,100,1]

# def solve_rec(i, cost, n):
#     if i >= n:
#         return 0
    
#     return cost[i] + min(solve_rec(i + 1, cost, n), solve_rec(i + 2, cost, n))
# def solve_mem(i, cost, n ,dp):
#     if i >= n:
#         return 0
#     if dp[i] != -1:
#         return dp[i]
#     return cost[i] + min(solve_mem(i + 1, cost, n, dp), solve_mem(i + 2, cost, n, dp))

# def solve_tab(cost, n):
#     dp = [-1] * (n + 1)
#     dp[0] = cost[0]
#     dp[1] = cost[1]
#     for i in range(2, n):
#             dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
#     return  min(dp[n-1], dp[n-2])

# def solve_SO(cost, n):
#     prev2 = cost[0]
#     prev1 = cost[1]
#     for i in range(2, n):
#             curr = min(prev1, prev2) + cost[i]
#             prev2 = prev1
#             prev1 = curr
#     return  min(prev2, prev1)

# def min_cost_climbing_stairs(cost):
#     n = len(cost)
#     # ans = min(solve_rec(0, cost, n), solve_rec(1, cost, n))
#     # dp = [-1] * (n + 1)
#     # ans = min(solve_mem(0, cost, n, dp), solve_mem(1, cost, n, dp))
#     # ans = solve_tab(cost, n)
#     ans = solve_SO(cost, n)
#     return ans

# print(min_cost_climbing_stairs(cost))

# # Minimum number of coins to make sum X
# # You are given an array of ‘N’ distinct integers and an integer ‘X’ representing the target sum. You have to tell the minimum number of elements you have to take to reach the target sum ‘X’.
# # Note:
# # You have an infinite number of elements of each type.

# coins = [1, 2, 6]
# target = 9
# def solve_rec(coins, target):
#     if target == 0:
#         return 0
#     if target < 0:
#         return float('inf')
#     min_ways = float('inf')
#     for coin in coins:
#         ans = solve_rec(coins, target- coin)
#         if ans != float('inf'):
#             min_ways = min(min_ways, 1 + ans)
#     return min_ways

# def solve_mem(coins, target, dp):
#     if target == 0:
#         return 0
#     if target < 0:
#         return float('inf')
#     if dp[target] != -1:
#         return dp[target]
#     min_ways = float('inf')
#     for coin in coins:
#         ans = solve_rec(coins, target - coin)
#         if ans != float('inf'):
#             min_ways = min(min_ways, 1 + ans)
#     dp[target] = min_ways
#     return dp[target]

# def solve_tab(coins, target):
#     dp = [float('inf')] * (target + 1)
#     dp[0] = 0
#     for i in range(1, target + 1):
#         for j in range(len(coins)):
#             if i - coins[j] >= 0 and dp[i - coins[j]] != float('inf'):
#                 dp[i] = min(dp[i], 1 + dp[i - coins[j]])
#     return dp[target]
    

# def minimum_no_of_coins(coins, target):
#     # ans = solve_rec(coins, target)
#     # dp = [-1] * (target + 1)
#     # ans = solve_mem(coins, target, dp)
#     ans = solve_tab(coins, target)
#     if ans == float('inf'):
#         return -1
#     else :
#         return ans
# print(minimum_no_of_coins(coins, target))


# # Maximum Sum of non adjacent elements
# arr = [2, 1, 4, 9]
# def solve_rec(arr, n):
#     if n < 0:
#         return 0
#     if n == 0:
#         return arr[0]
#     include = solve_rec(arr, n - 2) + arr[n]
#     exclude = solve_rec(arr, n - 1) + 0
#     return max(include, exclude)
# def solve_mem(arr, n, dp):
#     if n < 0:
#         return 0
#     if n == 0:
#         return arr[0]
#     if dp[n] != -1:
#         return dp[n]
#     include = solve_mem(arr, n - 2, dp) + arr[n]
#     exclude = solve_mem(arr, n - 1, dp) + 0
#     dp[n] = max(include, exclude)
#     return dp[n]
# def solve_tab(arr):
#     n = len(arr)
#     if n == 0:
#         return 0
#     if n == 1:
#         return arr[0]
#     dp = [-1] * (n + 1)
#     dp[0] = arr[0]
#     dp[1] = max(dp[0] , arr[1])
#     for i in range(2, n):
#         include = dp[i - 2] + arr[i]
#         exclude = dp[i - 1] 
#         dp[i] = max(include, exclude)
#     return dp[n-1]

# def solve_SO(arr):
#     n = len(arr)
#     # dp = [-1] * (n + 1)
#     prev2 = 0
#     prev1 = arr[0]
#     for i in range(1, n):
#         include = prev2+ arr[i]
#         exclude = prev1 
#         curr = max(include, exclude)
#         prev2 = prev1
#         prev1 = curr
#     return prev1

# def maximum_sum_non_adjacent_elements(arr):
#     n = len(arr)
#     if n == 0:
#         return 0
#     if n == 1:
#         return arr[0]
#     # ans = solve_rec(arr, n - 1) 
#     # dp = [-1] * (n + 1)
#     # ans = solve_mem(arr, n - 1, dp)
#     # ans = solve_tab(arr)
#     ans = solve_SO(arr)
#     return ans

# print(maximum_sum_non_adjacent_elements(arr))

# # house robbery problem if first and last house are also adjacent

# arr = [8,9,5,2,6]
# def solve_rec(arr, n):
#     if n == 0:
#         return arr[0]
#     if n < 0:
#         return 0
#     include = solve_rec(arr, n - 2) + arr[n]
#     exclude = solve_rec(arr, n - 1) + 0
#     return max(include, exclude)
# def solve_SO(arr):
#     n = len(arr)
#     prev2 = 0
#     prev1 = arr[0]
#     for i in range(1, n):
#         include  = prev2 + arr[i]
#         exclude = prev1
#         curr = max(include, exclude)
#         prev2 = prev1
#         prev1 = curr
#     return prev1
# def circular_house_robbery(arr):
#     n = len(arr)
#     if n == 1:
#         return arr[0]
#     exclude_first = []
#     exclude_last = []
#     for i in range(len(arr)):
#         if i != 0 : #exclude first
#             exclude_first.append(arr[i])
#         if i != len(arr) - 1 : # exclude last
#             exclude_last.append(arr[i])

#     # ans = max(solve_rec(exclude_first,n-2), solve_rec(exclude_last,n-2))
#     ans = max(solve_SO(exclude_first), solve_SO(exclude_last))
#     return ans

# print(circular_house_robbery(arr))



# # Cut rod into segements 
# #Given a rod of length L, the task is to cut the rod in such a way that the total number of segments of length p, q, and r is maximized. The segments can only be of length p, q, and r. 

# N = 7
# x = 4
# y = 2
# z = 3
# def solve_rec(n , x, y, z):
#     if n == 0:
#         return 0
#     if n < 0:
#         return float('-inf')
#     count_from_x = 1 + solve_rec(n - x, x , y, z)
#     count_from_y = 1 + solve_rec(n - y, x , y, z)
#     count_from_z = 1 + solve_rec(n - z, x , y, z)

#     return max(count_from_x, count_from_y, count_from_z)

# def solve_mem(n , x, y, z, dp):
#     if n == 0:
#         return 0
#     if n < 0:
#         return float('-inf')
#     if dp[n] != -1:
#         return dp[n]
#     count_from_x = 1 + solve_mem(n - x, x , y, z, dp)
#     count_from_y = 1 + solve_mem(n - y, x , y, z, dp)
#     count_from_z = 1 + solve_mem(n - z, x , y, z, dp)

#     dp[n] = max(count_from_x, count_from_y, count_from_z)
#     return dp[n]

# def solve_tab(n, x, y, z):
#     dp = [float('-inf')] * (n + 1)
#     dp[0] = 0

#     for i in range(1, n + 1):
#         if i - x >= 0:
#             dp[i] = max(dp[i], 1 + dp[i-x])
#         if i - y >= 0:
#             dp[i] = max(dp[i], 1 + dp[i-y])
#         if i - z >= 0:
#             dp[i] = max(dp[i], 1 + dp[i-z])

#     return dp[n]


# def cut_rod_into_segements(n, x, y, z):
#     # ans = solve_rec(n, x, y, z)
#     # dp = [-1] * (n + 1)
#     # ans = solve_mem(n, x, y, z, dp)
#     ans = solve_tab(n, x, y, z)
#     return ans

# print(cut_rod_into_segements(N, x, y, z))


# count dearangements
# A Derangement is a permutation of n elements, such that no element appears in its original position. For example, a derangement of {0, 1, 2, 3} is {2, 3, 1, 0}.
# Given a number n, find the total number of Derangements of a set of n elements.
# f(n) = n-1*(f(n-1)+f(n-2)) -> f(n-2) means we  swapped position of two elements (0 and i ) and f(n-1) means we palce 0 at i index and dont want i to be palced at 0 index

# n = 2
# def solve_rec(n):
#     if n == 1:
#         return 0
#     if n == 2:
#         return 1
#     ans =  (n - 1) * (solve_rec(n - 1) + solve_rec(n - 2)) 
#     return ans
# def solve_mem(n, dp):
#     if n == 1:
#         return 0
#     if n == 2:
#         return 1
#     if dp[n] != -1:
#         return dp[n]
#     dp[n] =  (n - 1) * (solve_mem(n - 1, dp) + solve_mem(n - 2, dp)) 
#     return dp[n]

# def solve_tab(n):
#     if n == 1:
#         return 0
#     dp = [-1] * (n + 1)
#     dp[1] = 0
#     dp[2] = 1
#     for i in range(3, n + 1):
#         dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])
#     return dp[n] 

# def solve_SO(n):
#     if n == 1:
#         return 0
#     prev2 = 0
#     prev1 = 1
#     for i in range(3, n + 1):
#         curr = (i - 1) * (prev2 + prev1)
#         prev2 = prev1
#         prev1 = curr
#     return prev1

# def count_dearangements(n):
#     # ans = solve_rec(n)
#     # dp = [-1] * (n + 1)
#     # ans = solve_mem(n, dp)
#     # ans = solve_tab(n)
#     ans = solve_SO(n)
#     return ans
# print(count_dearangements(n))

# #Painting fence algorithm
# #Given a fence with n posts and k colors, find out the number of ways of painting the fence so that not more than two consecutive posts have the same colors.
# for base case n == 2 we have two conditions -> when both fence have same color (k ways) + when both fence have different color (k)*(k-1)
#  for recursive function if last two fence have same color then (k-1) * f(n - 2) + if last fence have different color from second last then (k - 1) * f(n-1)

# n = 3
# k = 2
# MOD = 10**9 + 7
# def solve_rec(n, k):
#     if n == 1:
#         return k
#     if n == 2:
#         return k + k * (k - 1) % MOD
#     ans = (k - 1) * solve_rec(n - 1, k) + (k - 1) * solve_rec(n - 2, k)
#     return ans

# def solve_mem(n, k, dp):
#     if n == 1:
#         return k
#     if n == 2:
#         return k + k * (k - 1) % MOD
#     if dp[n] != -1:
#         return dp[n]
#     dp[n] = (k - 1) * solve_rec(n - 1, k) + (k - 1) * solve_rec(n - 2, k)
#     return dp[n]
    
# def solve_tab(n, k):
#     if  n == 1:
#         return k
#     if n == 2:
#         return (k * k) % MOD
#     dp = [-1] * (n + 1)
#     dp[1] = k
#     dp[2] = (k * k) % MOD
#     for i in range(3, n + 1):
#         # dp[i] = (k - 1) * dp[n-1] + (k - 1) * dp[n - 2]
#         dp[i] = ((k - 1) * (dp[n-1] + dp[n - 2])) % MOD
#     return dp[n]

# def solve_SO(n, k):
#     if  n == 1:
#         return k
#     if n == 2:
#         return (k * k) % MOD
#     prev2 = k
#     prev1 = k * k
#     for _ in range(3, n + 1):
#         # curr = (k - 1) * prev1 + (k - 1) * prev2
#         curr = ((k - 1) * (prev1 + prev2)) % MOD
#         prev2 = prev1
#         prev1 = curr
#     return prev1


# def painting_fence(n, k):
#     # ans = solve_rec(n, k)
#     # dp = [-1] * (n + 1)
#     # ans = solve_mem(n, k, dp)
#     # ans = solve_tab(n, k)
#     ans = solve_SO(n, k)
#     return ans
# print(painting_fence(n, k))


# # combination sum IV 
# #Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.
# nums = [1,2,3]
# target = 4
# def solve_rec(nums, target):
#     if target == 0:
#         return 1
#     if target < 0:
#         return  0
#     combinations = 0
#     for num in nums:
#         ans = solve_rec(nums, target - num)
#         combinations += ans
#     return combinations

# def solve_mem(nums, target, dp):
#     if target == 0:
#         return 1
#     if target < 0:
#         return  0
#     if dp[target] != -1:
#         return dp[target]
#     combinations = 0
#     for num in nums:
#         ans = solve_mem(nums, target - num, dp)
#         combinations += ans
#     dp[target] = combinations
#     return dp[target]
# def solve_tab(nums, target):
#     dp = [0 for _ in range(target + 1)]
#     dp[0] = 1
#     for i in range(1 ,target + 1):
#         for num in nums:
#             if i - num >= 0:
#                 dp[i] += dp[i - num]
#     return dp[target]


# def combination_sum_IV(nums, target):
#     # ans = solve_rec(nums, target)
#     # dp = [-1] * (target + 1)
#     # ans = solve_mem(nums, target, dp)
#     ans = solve_tab(nums, target)
#     return ans
# print(combination_sum_IV(nums, target))


# # perfect square problem 
# # find the minimum no of squares of any number that sum to N
# n = 6
# def solve_rec(n):
#     if n == 0:
#         return 0
#     min_ans = n # worst case is that if the number is formed from squares of all 1
#     ans = 0
#     for i in range(1, n + 1):
#         if i * i > n:
#             break
#         ans = 1 + solve_rec(n - i * i)
#         min_ans = min(min_ans, ans)
#     return min_ans

# def solve_mem(n, dp):
#     if n == 0:
#         return 0
#     min_ans = n
#     ans = 0
#     if dp[n] != -1:
#         return dp[n]
#     for i in range(1, n + 1):
#         if i * i > n:
#             break
#         ans = 1 + solve_mem(n - i * i, dp)
#         min_ans = min(min_ans, ans)
#     dp[n] = min_ans
#     return dp[n]

# def solve_tab(n):
#     dp = [float('inf') for _ in range(n + 1)]
#     dp[0] = 0
#     for j in range(1, n + 1):
#         for i in range(1, n + 1):
#             if i * i > j:
#                 break
#             if j - i * i >= 0:
#                 dp[j] = min(dp[j], 1 + dp[j - i * i])
#     return dp[n]

# def prefect_square_problem(n):
#     # ans = solve_rec(n)
#     # dp = [-1] * (n + 1)
#     # ans = solve_mem(n, dp)
#     ans = solve_tab(n)
#     return ans
# print(prefect_square_problem(n))


# 2d dp questions with pattern inculde/exclude
# 0/1 knapsack
# equalsubset sum partition
# subset sum
# minimum subset sum differemce 
# count of subset sum
# target sum

# #0/1knapsack problem
# n = 4
# weight = [1, 2, 4, 5]
# value = [5, 4, 8, 6]
# max_weight = 5

# def solve_rec(weight, value, max_weight, index):
#     if index  == 0:
#         if weight[0] <= max_weight:
#             return value[0]
#         else:
#             return 0
#     include = 0
#     if weight[index] <= max_weight:
#         include = value[index] + solve_rec(weight, value, max_weight - weight[index], index - 1)
#     exclude = solve_rec(weight, value, max_weight, index - 1)

#     ans = max(include, exclude)
#     return ans

# def solve_mem(weight, value, max_weight, index, dp):
#     if index  == 0:
#         if weight[0] <= max_weight:
#             return value[0]
#         else:
#             return 0
#     if dp[index][max_weight] != -1:
#         return dp[index][max_weight]
#     include = 0
#     if weight[index] <= max_weight:
#         include = value[index] + solve_mem(weight, value, max_weight - weight[index], index - 1, dp)
#     exclude = solve_mem(weight, value, max_weight, index - 1, dp)

#     dp[index][max_weight] = max(include, exclude)
#     return dp[index][max_weight]

# def solve_tab(weight, value, max_weight, index):
#     dp = [[-1 for _ in range(max_weight + 1)]for _ in range(index + 1)]

#     for w in range(max_weight + 1):
#         if weight[0] <= w:
#             dp[0][w] = value[0]
#         else:
#             dp[0][w] = 0
    
#     for i in range(1, index):
#         for w in range(max_weight + 1):
#             include = 0
#             if weight[i] <= w:
#                 include = value[i] + dp[i - 1][w - weight[i]]
#             exclude = dp[i - 1][w]
#             dp[i][w] = max(include, exclude)
    
#     return dp[index - 1][max_weight]


# def solve_SO(weight, value, max_weight, index):
#     prev = [-1 for _ in range(max_weight + 1)]
#     curr = [-1 for _ in range(max_weight + 1)]

#     for w in range(max_weight + 1):
#         if weight[0] <= w:
#             prev[w] = value[0]
#         else:
#             prev[w] = 0
    
#     for i in range(1, index):
#         for w in range(max_weight + 1):
#             include = 0
#             if weight[i] <= w:
#                 include = value[i] + prev[w - weight[i]]
#             exclude = prev[w]
#             curr[w] = max(include, exclude)
#         prev = curr[:]
    
#     return prev[max_weight]

# def solve_one_array(weight, value, max_weight, index):
#     curr = [-1 for _ in range(max_weight + 1)]

#     for w in range(max_weight + 1):
#         if weight[0] <= w:
#             curr[w] = value[0]
#         else:
#             curr[w] = 0
    
#     # just have to reverse the loop of w (second loop)
#     for i in range(1, index):
#         for w in range(max_weight, -1, -1):
#             include = 0
#             if weight[i] <= w:
#                 include = value[i] + curr[w - weight[i]]
#             exclude = curr[w]
#             curr[w] = max(include, exclude)
    
#     return curr[max_weight]

    
# def solve_knapsack(weight, value, max_weight, n):
#     # ans = solve_rec(weight, value, max_weight, n - 1)
#     # dp = [[-1 for _ in range(max_weight + 1)]for _ in range(n + 1)]
#     # ans = solve_mem(weight, value, max_weight, n - 1, dp)
#     # ans = solve_tab(weight, value, max_weight, n)
#     # ans = solve_SO(weight, value, max_weight, n)
#     ans = solve_one_array(weight, value, max_weight, n)
#     return ans
# print(solve_knapsack(weight, value, max_weight, n))


# # Maximal Square
# # Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's

# n, m = 3, 4 
# matrix = [[0, 1, 1, 1],
#           [1, 1, 1, 0],
#           [0, 0, 1, 0]
#           ]
# # matrix = [[1, 1],
# #           [0, 1]
# #           ]
# def solve_rec(matrix, i, j, maxi):
#     if i >= len(matrix) or j >= len(matrix[0]):
#         return 0

#     right = solve_rec(matrix, i, j + 1, maxi)
#     diagonal = solve_rec(matrix, i + 1, j + 1, maxi)
#     down = solve_rec(matrix, i + 1, j, maxi)

#     ans = 0
#     if matrix[i][j] == 1:
#         ans = 1 + min(right, diagonal, down)
#         maxi[0] = max(maxi[0], ans)
#         return ans
#     else :
#         return 0
    
# def solve_mem(matrix, i, j, maxi, dp):
#     if i >= len(matrix) or j >= len(matrix[0]):
#         return 0
#     if dp[i][j] != -1:
#         return dp[i][j]
#     right = solve_mem(matrix, i, j + 1, maxi, dp)
#     diagonal = solve_mem(matrix, i + 1, j + 1, maxi, dp)
#     down = solve_mem(matrix, i + 1, j, maxi, dp)

#     ans = 0
#     if matrix[i][j] == 1:
#         ans = 1 + min(right, diagonal, down)
#         maxi[0] = max(maxi[0], ans)
#         dp[i][j] =  ans
#         return dp[i][j]
#     else :
#         dp[i][j] =  0
#         return dp[i][j]

# def solve_tab(matrix, maxi):
#     n = len(matrix)
#     m = len(matrix[0])
#     dp = [[0 for _ in range(m + 1)]for _ in range(n + 1)]
#     for i in range(n-1, -1, -1):
#         for j in range(m-1, -1, -1):
#             right = dp[i][j + 1]
#             diagonal =  dp[i + 1][j + 1]
#             down =  dp[i+ 1][j]
#             if matrix[i][j] == 1:
#                 dp[i][j] = 1 + min(right, diagonal, down)
#                 maxi[0] = max(maxi[0], dp[i][j])
#             else:
#                 dp[i][j] = 0
#     return maxi[0]

# def solve_SO(matrix, maxi):
#     n = len(matrix)
#     m = len(matrix[0])
#     # dp = [[0 for _ in range(m + 1)]for _ in range(n + 1)]
#     curr = [0 for _ in range(m + 1)]
#     next = [0 for _ in range(m + 1)]
#     for i in range(n-1, -1, -1):
#         for j in range(m-1, -1, -1):
#             right = curr[j + 1]
#             diagonal =  next[j + 1]
#             down =  next[j]
#             if matrix[i][j] == 1:
#                 curr[j] = 1 + min(right, diagonal, down)
#                 maxi[0] = max(maxi[0], curr[j])
#             else:
#                 curr[j] = 0
#         next = curr[:]
#     return maxi[0]

    
# def maximal_square(matrix):
#     maxi = [0]
#     # ans = solve_rec(matrix,0, 0, maxi)
#     # dp = [[-1 for _ in range(len(matrix[0])+ 1)]for _ in range(len(matrix) + 1)]
#     # solve_mem(matrix, 0, 0, maxi, dp)
#     # ans = solve_tab(matrix, maxi)
#     ans = solve_SO(matrix, maxi)
#     return ans
#     # return maxi[0]

# print(maximal_square(matrix))


# # Minimum score triangulation
# arr = [3, 7, 4, 5]
# def solve_rec(arr, i, j):
#     if i + 1 == j:
#         return 0
#     ans = float('inf')
#     for k in range(i + 1, j):
#         ans = min(ans, arr[i] * arr[j] * arr[k] + solve_rec(arr, i, k)+ solve_rec(arr, k, j))
#     return ans
# def solve_mem(arr, i, j, dp):
#     if i + 1 == j:
#         return 0
#     if dp[i][j] != -1:
#         return dp[i][j]
#     ans = float('inf')
#     for k in range(i + 1, j):
#         ans = min(ans, arr[i] * arr[j] * arr[k] + solve_mem(arr, i, k, dp)+ solve_mem(arr, k, j, dp))
#     dp[i][j] =  ans
#     return dp[i][j]
# def solve_tab(arr):
#     n = len(arr)
#     dp = [[0 for _ in range(n + 1)]for _ in range(n + 1)]
#     for i in range(n - 1, -1, -1):
#         for j in range(i + 2, n):
#             ans = float('inf')
#             for k in range(i + 1, j, 1):
#                 ans = min(ans, arr[i] * arr[j] * arr[k] +  dp[i][k] + dp[k][j])
#             dp[i][j] = ans
#     return dp[0][n - 1]

# def minimum_score_triangulation(arr):
#     n = len(arr)
#     # ans = solve_rec(arr, 0, n - 1)
#     # dp = [[-1 for _ in range(n + 1)]for _ in range(n + 1)]
#     # ans = solve_mem(arr, 0, n - 1, dp)
#     ans = solve_tab(arr)
#     return ans
# print(minimum_score_triangulation(arr))


# # Reducing dishes
# satisfaction = [-1, -8, 0, 5, -9]

# def solve_rec(arr, index, time):
#     if index  == len(arr):
#         return 0
#     include = arr[index] * (time + 1) + solve_rec(arr, index + 1, time + 1)
#     exclude = 0 + solve_rec(arr, index + 1, time)
#     return max(include, exclude)

# def solve_mem(arr, index, time, dp):
#     if index  == len(arr):
#         return 0
#     if dp[index][time] != -1:
#         return dp[index][time]
#     include = arr[index] * (time + 1) + solve_mem(arr, index + 1, time + 1, dp)
#     exclude = 0 + solve_mem(arr, index + 1, time, dp)
#     dp[index][time] =  max(include, exclude)
#     return dp[index][time]

# def solve_tab(arr):
#     n = len(arr)
#     dp = [[0 for _ in range(n + 1)]for _ in range(n + 1)]
#     for index in range(n-1, -1 ,-1):
#         for time in range(index, -1, -1):
#             include = arr[index] * (time + 1) + dp[index + 1][time + 1]
#             exclude = 0 + dp[index + 1][time]
#             dp[index][time] = max(include, exclude)
#     return dp[0][0]
# def solve_SO(arr):
#     n = len(arr) 
#     # dp = [[0 for _ in range(n + 1)]for _ in range(n + 1)]
#     curr = [0 for _ in range(n + 1)]
#     next = [0 for _ in range(n + 1)]
    
#     for index in range(n-1, -1, -1):
#         for time in range(index, -1, -1):
#             include = arr[index] * (time + 1) + next[time + 1]
#             exclude = 0 + next[time]
#             curr[time] = max(include, exclude)
#         next = curr[:]
#     return next[0]
# def reducing_dishes(arr):
#     arr.sort()
#     # ans = solve_rec(arr, 0, 0)
#     # dp = [[-1 for _ in range(len(arr) + 1)]for _ in range(len(arr) + 1)]
#     # ans = solve_mem(arr, 0, 0, dp)
#     # ans =  solve_tab(arr)
#     ans = solve_SO(arr)
#     return ans
# print(reducing_dishes(satisfaction))


# # Minimum cost for tickets
# days = [1,4,6,7,8,20]
# costs = [2,7,15]
# def solve_rec(days, costs, index):
#     if index >= len(days):
#         return 0
#     # one day pass
#     option1 = costs[0] + solve_rec(days, costs, index + 1)
#     # 7 days pass
#     i = index
#     while i < len(days) and days[i] < days[index] + 7:
#         i += 1
#     option2 = costs[1] + solve_rec(days, costs, i)
#     # 30 days pass
#     i = index
#     while i < len(days) and days[i] < days[index] + 30:
#         i += 1
#     option3 = costs[2] + solve_rec(days, costs, i)

#     return min(option1, option2, option3)

# def solve_mem(days, costs, index, dp):
#     if index >= len(days):
#         return 0
#     if dp[index] != -1:
#         return dp[index]
#     option1 = costs[0] + solve_mem(days, costs, index + 1, dp)

#     i = index
#     while i < len(days) and days[i] < days[index] + 7:
#         i += 1
#     option2 = costs[1] + solve_mem(days, costs, i, dp)

#     i = index 
#     while i < len(days) and days[i] < days[index] + 30:
#         i += 1
#     option3 = costs[2] + solve_mem(days, costs, i, dp)

#     dp[index] = min(option1, option2, option3)
#     return dp[index] 
# def solve_tab(days, costs):
#     n = len(days)
#     dp = [float('inf') for _ in range(n + 1)]
#     dp[n] = 0
#     for k in range(n-1, -1, -1):
#         option1 = costs[0] + dp[k + 1]

#         i = k
#         while i < len(days) and days[i] < days[k] + 7:
#             i += 1
#         option2 = costs[1] + dp[i]

#         i = k 
#         while i < len(days) and days[i] < days[k] + 30:
#             i += 1
#         option3 = costs[2] + dp[i]

#         dp[k] = min(option1, option2, option3)
#     return dp[0]

# def minimum_cost_for_tickets(days, costs):
#     # ans = solve_rec(days, costs, 0)
#     # dp = [-1 for _ in range(len(days) + 1)]
#     # ans = solve_mem(days, costs, 0, dp)
#     ans = solve_tab(days, costs)
#     return ans
# print(minimum_cost_for_tickets(days, costs))


# # Longest Increasing Subsequence
# nums = [10,9,2,5,3,7,101,18]
# # nums = [9, 2, 5, 3]
# def solve_rec(arr, curr, prev):
#     if curr == len(arr):
#         return 0
#     include = 0
#     if prev == -1 or arr[curr] > arr[prev]:
#         include = 1 + solve_rec(arr, curr + 1, curr)
#     exclude = 0 + solve_rec(arr, curr + 1, prev)
#     return max(include, exclude)

# def solve_mem(arr, curr, prev, dp):
#     if curr == len(arr):
#         return 0
#     if dp[curr][prev]  != -1:
#         return dp[curr][prev] 
#     include = 0
#     if prev == -1 or arr[curr] > arr[prev]:
#         include = 1 + solve_mem(arr, curr + 1, curr, dp)
#     exclude = 0 + solve_mem(arr, curr + 1, prev, dp)
#     dp[curr][prev] = max(include, exclude)
#     return dp[curr][prev]

# def solve_tab(arr):
#     n = len(arr)
#     dp = [[0 for _ in range(n + 1)]for _ in range(n + 1)]

#     for curr in range(n-1, -1, -1):
#         for prev in range(curr - 1, -2, -1):
#             include = 0
#             if prev == -1 or arr[curr] > arr[prev]:
#                 include = 1 + dp[curr + 1][curr]
#             exclude = 0 + dp[curr + 1][prev]
#             dp[curr][prev] = max(include, exclude)
#     return dp[0][-1]

# def solve_SO(arr):
#     n = len(arr)
#     # dp = [[0 for _ in range(n + 1)]for _ in range(n + 1)]
#     curr_row = [0 for _ in range(n + 1)]
#     next_row = [0 for _ in range(n + 1)]

#     for curr in range(n-1, -1, -1):
#         for prev in range(curr - 1, -2, -1):
#             include = 0
#             if prev == -1 or arr[curr] > arr[prev]:
#                 include = 1 + next_row[curr]
#             exclude = 0 + next_row[prev]
#             curr_row[prev] = max(include, exclude)
#         next_row = curr_row[:]
#     return curr_row[-1]
# def find_lower_bound(arr, element):
#     left = 0
#     right = len(arr)
#     while left < right:
#         mid = (left + right) // 2
#         if arr[mid] < element:
#             left = mid + 1
#         else :
#             right  = mid
#     return left
# def solve_dp_with_binary_search(nums):
#     n = len(nums)
#     ans_lst = []
#     ans_lst.append(nums[0])
#     for i in range(1, n):
#         if nums[i] > ans_lst[-1]:
#             ans_lst.append(nums[i])
#         else:
#             index  = find_lower_bound(ans_lst, nums[i])
#             ans_lst[index] = nums[i]
#     return len(ans_lst)

# def longest_increasing_subsequence(nums):
#     # ans = solve_rec(nums, 0, -1)
#     # n = len(nums)
#     # dp = [[-1 for _ in range(n + 1)]for _ in range(n + 1)]
#     # ans = solve_mem(nums, 0, -1, dp)
#     # ans = solve_tab(nums)
#     # ans = solve_SO(nums)
#     ans = solve_dp_with_binary_search(nums)
#     return ans
# print(longest_increasing_subsequence(nums))

# # Minimum Sideways jump 
# # obstacles = [0,1,2,3,0]
# obstacles = [0,1,1,3,3,0]
# def solve_rec(obstacles, currlane, currpos):
#     if currpos == len(obstacles) - 1:
#         return 0
#     if obstacles[currpos + 1] != currlane:
#         return solve_rec(obstacles, currlane, currpos + 1)
#     else:
#         # sideways jump 
#         min_sideways = float('inf')
#         for i in range(1, 4):
#             if currlane != i and obstacles[currpos] != i:
#                 min_sideways = min(min_sideways, 1 + solve_rec(obstacles, i, currpos))
#     return min_sideways

# def solve_mem(obstacles, currlane, currpos, dp):
#     if currpos == len(obstacles) - 1:
#         return 0
#     if dp[currlane][currpos] != -1:
#         return dp[currlane][currpos]
#     if obstacles[currpos + 1] != currlane:
#         return  solve_mem(obstacles, currlane, currpos + 1, dp)
#     else:
#         # sideways jump
#         min_sidways = float('inf')
#         for i in range(1, 4):
#             if currlane != i and obstacles[currpos] != i:
#                 min_sidways = min(min_sidways, 1 + solve_mem(obstacles, i, currpos, dp))
#     dp[currlane][currpos] = min_sidways
#     return dp[currlane][currpos]
    
# def solve_tab(obstacles):
#     n = len(obstacles)
#     dp = [[float('inf') for _ in range(n)]for _ in range(4)]
#     dp[0][n - 1] = 0
#     dp[1][n - 1] = 0
#     dp[2][n - 1] = 0
#     dp[3][n - 1] = 0

#     for currpos in range(n - 2, -1 ,-1):
#         for currlane in range(1, 4):
#             if obstacles[currpos + 1] != currlane:
#                 dp[currlane][currpos]  = dp[currlane][currpos + 1]
#             else:
#                 # sideways jump
#                 min_sidways = float('inf')
#                 for i in range(1, 4):
#                     if currlane != i and obstacles[currpos] != i:
#                         min_sidways = min(min_sidways, 1 + dp[i][currpos + 1])
#                 dp[currlane][currpos] = min_sidways
#     return min(dp[2][0], dp[1][0] + 1, dp[3][0] + 1)
# def solve_SO(obstacles):
#     n = len(obstacles)
#     curr = [float('inf') for _ in range(4)]
#     next = [float('inf') for _ in range(4)]
#     next[0] = 0
#     next[1] = 0
#     next[2] = 0
#     next[3] = 0
#     for currpos in range(n-2, -1, -1):
#         for currlane in range(1, 4):
#             if obstacles[currpos + 1] != currlane:
#                 curr[currlane]  = next[currlane]
#             else:
#                 # sideways jump
#                 min_sidways = float('inf')
#                 for i in range(1, 4):
#                     if currlane != i and obstacles[currpos] != i:
#                         min_sidways = min(min_sidways, 1 + next[i])
#                 curr[currlane] = min_sidways
#         next = curr[:]
#     return min(next[2], next[1] + 1, next[3] + 1)

# def minimum_sideways_jump(obstacles):
#     # ans = solve_rec(obstacles, 2, 0)
#     # dp = [[-1 for _ in range(len(obstacles))]for _ in range(4)]
#     # ans = solve_mem(obstacles, 2, 0, dp)
#     # ans = solve_tab(obstacles)
#     ans = solve_SO(obstacles)
#     return ans
# print(minimum_sideways_jump(obstacles))


# # Russian DOll
# # You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width and the height of an envelope.
# # One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.
# # Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).
# envelopes = [[5,4],[6,4],[6,7],[2,3]]
# def find_lower_bound(arr, element):
#     left = 0
#     right = len(arr)
#     while(left < right):
#         mid = (left + right)//2
#         if arr[mid] < element:
#             left = mid + 1
#         else:
#             right  = mid
#     return left

# def solve_dp_with_binary_search(envelopes):
#     # First sort ascending based on first column(width) and if two envelopes consist of same width then sort them in desc based on height
#     # then find the length of LIS for second column(height)
#     # sorted_arr = sorted(envelopes, key= lambda x : (x[0], -x[1]))
#     sorted_arr = sorted(envelopes, key = lambda x : x[1], reverse = True)
#     sorted_arr = sorted(sorted_arr, key = lambda x : x[0])
#     ans_lst = []
#     ans_lst.append(sorted_arr[0][1])
#     for i in range(1, len(sorted_arr)):
#         if sorted_arr[i][1] > ans_lst[-1]:
#             ans_lst.append(sorted_arr[i][1])
#         else:
#             index  = find_lower_bound(ans_lst, sorted_arr[i][1])
#             ans_lst[index] = sorted_arr[i][1]
#     return len(ans_lst)
# def russian_doll_enevelope(envelopes):
#     ans = solve_dp_with_binary_search(envelopes)
#     return ans
# print(russian_doll_enevelope(envelopes))

# # Maximum Height by Stacking cuboid
# # Given n cuboids where the dimensions of the ith cuboid is cuboids[i] = [widthi, lengthi, heighti] (0-indexed). Choose a subset of cuboids and place them on each other.
# # You can place cuboid i on cuboid j if widthi <= widthj and lengthi <= lengthj and heighti <= heightj. You can rearrange any cuboid's dimensions by rotating it to put it on another cuboid.
# # Return the maximum height of the stacked cuboids.
# cuboids = [[50,45,20],[95,37,53],[45,23,12]]
# def check(curr_cuboid, prev_cuboid):
#         if curr_cuboid[0] >= prev_cuboid[0] and curr_cuboid[1] >= prev_cuboid[1] and curr_cuboid[2] >= prev_cuboid[2]:
#             return True
#         return False
        
# def solve_using_LIS_mem(arr, curr, prev, dp):
#     if curr == len(arr):
#         return 0
#     include = 0
#     if dp[curr][prev] != -1:
#         return dp[curr][prev]
#     if prev == -1 or check(arr[curr], arr[prev]):
#         include = arr[curr][2] + solve_using_LIS_mem(arr, curr + 1, curr, dp)
#     exclude = solve_using_LIS_mem(arr, curr + 1, prev, dp)
#     dp[curr][prev] = max(include, exclude)
#     return dp[curr][prev]

# def solve_using_LIS_tab(arr):
#     n = len(arr)
#     dp = [[0 for _ in range(n + 1)]for _ in range(n + 1)]
#     for curr in range(n - 1, -1 , -1):
#         for prev in range(curr -1 , -2, -1):
#             include = 0
#             if prev == -1 or check(arr[curr], arr[prev]):
#                 include = arr[curr][2] + dp[curr + 1][curr]
#             exclude = dp[curr + 1][prev]
#             dp[curr][prev] = max(include, exclude)
#     return dp[0][-1]

# def solve_using_LIS_SO(arr):
#     n = len(arr)
#     # dp = [[0 for _ in range(n + 1)]for _ in range(n + 1)]
#     curr_row = [0 for _ in range(n + 1)]
#     next_row = [0 for _ in range(n + 1)]
#     for curr in range(n - 1, -1 , -1):
#         for prev in range(curr -1 , -2, -1):
#             include = 0
#             if prev == -1 or check(arr[curr], arr[prev]):
#                 include = arr[curr][2] + next_row[curr]
#             exclude = next_row[prev]
#             curr_row[prev] = max(include, exclude)
#         next_row = curr_row[:]
#     return next_row[-1]

# def maximum_height_by_stacking_cuboid(cuboids):
#     # sort all rows, by doing this we will consider last value of each row as height of cuboid
#     # now sort 2d matrix based on their first value . we assume that as width
#     # use LIS to find maximum height
#     #step1
#     for row in cuboids:
#         row.sort()
#     # step2
#     sorted_cuboid = sorted(cuboids, key = lambda x : (x[0], x[1], x[2]))
#     # step3
#     # n = len(sorted_cuboid)
#     # dp = [[-1 for _ in range(n + 1)]for _ in range(n + 1)]
#     # ans = solve_using_LIS_mem(sorted_cuboid, 0, -1, dp)
#     # ans = solve_using_LIS_tab(sorted_cuboid)
#     ans = solve_using_LIS_SO(sorted_cuboid)
#     return ans
# print(maximum_height_by_stacking_cuboid(cuboids))


# # Pizza With 3n Slices
# #There is a pizza with 3n slices of varying size, you and your friends will take slices of pizza as follows:
# # You will pick any pizza slice.
# # Your friend Alice will pick the next slice in the anti-clockwise direction of your pick.
# # Your friend Bob will pick the next slice in the clockwise direction of your pick.
# # Repeat until there are no more slices of pizzas.
# # Given an integer array slices that represent the sizes of the pizza slices in a clockwise direction, return the maximum possible sum of slice sizes that you can pick.

# slices = [1,2,3,4,5,6]
# def solve_rec(arr, index, slices_to_eat, end_index):
#     if slices_to_eat == 0 or index > end_index:
#         return 0
#     include = arr[index] + solve_rec(arr, index + 2, slices_to_eat - 1, end_index)
#     exclude = 0 + solve_rec(arr, index + 1, slices_to_eat, end_index)
#     return max(include, exclude)

# def solve_mem(arr, index, slices_to_eat, end_index, dp):
#     if slices_to_eat == 0 or index > end_index:
#         return 0
#     if dp[index][slices_to_eat] != -1:
#         return dp[index][slices_to_eat]
#     include = arr[index] + solve_mem(arr, index + 2, slices_to_eat - 1, end_index, dp)
#     exclude = 0 + solve_mem(arr, index + 1, slices_to_eat, end_index, dp)
#     dp[index][slices_to_eat] =  max(include, exclude)
#     return dp[index][slices_to_eat]
# def solve_tab(arr):
#     n = len(arr)
#     no_of_slices_for_each = n // 3
#     dp1 = [[0 for _ in range(no_of_slices_for_each + 1)]for _ in range(n + 2)]
#     dp2 = [[0 for _ in range(no_of_slices_for_each + 1)]for _ in range(n + 2)]

#     for index in range(n - 2, -1, -1):
#         for slices_to_eat in range(1, no_of_slices_for_each + 1, 1):
#             include = arr[index] + dp1[index + 2][slices_to_eat - 1]
#             exclude = 0 + dp1[index + 1][slices_to_eat]
#             dp1[index][slices_to_eat] =  max(include, exclude)
#     with_first_element = dp1[0][no_of_slices_for_each]

#     for index in range(n - 1, 0, -1):
#         for slices_to_eat in range(1, no_of_slices_for_each + 1, 1):
#             include = arr[index] + dp2[index + 2][slices_to_eat - 1]
#             exclude = 0 + dp2[index + 1][slices_to_eat]
#             dp2[index][slices_to_eat] =  max(include, exclude)
#     without_first_element = dp2[1][no_of_slices_for_each]

#     return max(with_first_element, without_first_element)

# def pizza_with_3n_slices(slices):
#     # n  = len(slices)
#     # slices_to_eat = n // 3
#     # with_first_element = solve_rec(slices, 0, slices_to_eat, n - 2)
#     # without_first_element = solve_rec(slices, 1, slices_to_eat, n - 1)
#     # dp1 = [[-1 for _ in range(slices_to_eat + 1)]for _ in range(n + 1)]
#     # with_first_element = solve_mem(slices, 0, slices_to_eat, n - 2, dp1)
#     # dp2 = [[-1 for _ in range(slices_to_eat + 1)]for _ in range(n + 1)]
#     # without_first_element = solve_mem(slices, 1, slices_to_eat, n - 1, dp2)
#     # ans = max(with_first_element, without_first_element)
#     ans = solve_tab(slices)
#     return ans
# print(pizza_with_3n_slices(slices))


# # Number of Dice Rolls With Target Sum
# # You have n dice, and each dice has k faces numbered from 1 to k.
# # Given three integers n, k, and target, return the number of possible ways (out of the kn total ways) to roll the dice, so the sum of the face-up numbers equals target. Since the answer may be too large, return it modulo 109 + 7.
# n, k = 2, 6
# target = 7
# def solve_rec(no_of_dice, no_of_faces, target):
#     if target < 0:
#         return 0
#     if no_of_dice != 0 and target  == 0:
#         return 0
#     if no_of_dice == 0 and target != 0:
#         return 0
#     if no_of_dice == 0 and target == 0:
#         return 1
#     ans = 0
#     for i in range(1, no_of_faces + 1):
#         ans = ans + solve_rec(no_of_dice - 1, no_of_faces, target - i)
#     return ans

# def solve_mem(no_of_dice, no_of_faces, target, dp):
#     if target < 0:
#         return 0
#     if no_of_dice != 0 and target  == 0:
#         return 0
#     if no_of_dice == 0 and target != 0:
#         return 0
#     if no_of_dice == 0 and target == 0:
#         return 1
#     if dp[no_of_dice][target] != -1:
#         return dp[no_of_dice][target]
#     ans = 0
#     for i in range(1, no_of_faces + 1):
#         ans = ans + solve_mem(no_of_dice - 1, no_of_faces, target - i, dp)
#     dp[no_of_dice][target] =  ans
#     return dp[no_of_dice][target]
# def solve_tab(no_of_dice, no_of_faces, target):
#     dp = [[0 for _ in range(target + 1)]for _ in range(no_of_dice + 1)]
#     dp[0][0] = 1
#     for dice in range(1, no_of_dice + 1):
#         for target_sum in range(1, target + 1):
#             ans = 0
#             for i in range(1, no_of_faces + 1):
#                 if target_sum - i >= 0:
#                     ans = ans + dp[dice - 1][target_sum - i]
#             dp[dice][target_sum] =  ans
#     return dp[no_of_dice][target]

# def dice_rolls_with_target_sum(no_of_dice, no_of_faces, target):
#     # ans = solve_rec(no_of_dice, no_of_faces, target)\
#     # dp = [[-1 for _ in range(target + 1)]for _ in range(no_of_dice + 1)]
#     # ans = solve_mem(no_of_dice, no_of_faces, target, dp)
#     ans = solve_tab(no_of_dice, no_of_faces, target)
#     return ans
    
# print(dice_rolls_with_target_sum(n , k, target))


# # Partition Equal SUBSET SUM
# # Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.

# nums = [1,5,11,7]
# def solve_rec(arr, index, target):
#     if target < 0:
#         return False
#     if target == 0:
#         return True
#     if index >= len(arr):
#         return False 
#     include = solve_rec(arr, index + 1, target - arr[index])
#     exclude = solve_rec(arr, index + 1, target)
#     return include or exclude
# def solve_mem(arr, index, target, dp):
#     if target < 0:
#         return False
#     if target == 0:
#         return True
#     if index >= len(arr):
#         return False 
#     if dp[index][target] != -1:
#         return dp[index][target]
#     include = solve_mem(arr, index + 1, target - arr[index], dp)
#     exclude = solve_mem(arr, index + 1, target, dp)
#     dp[index][target] = include or exclude
#     return dp[index][target]
# def solve_tab(arr, target):
#     n = len(arr)
#     dp = [[False for _ in range(target + 1)]for _ in range(n + 1)]
#     for i in range(n + 1):
#         dp[i][0] = True
    
#     for index in range(n - 1, -1, -1):
#         for target_sum in range(1, target + 1):
#             include = False
#             if target_sum - arr[index] >= 0:
#                 include = dp[index + 1][target_sum - arr[index]]
#             exclude = dp[index + 1][target_sum]
#             dp[index][target_sum] = include or exclude
#     return dp[0][target]

# def solve_SO(arr, target):
#     n = len(arr)
#     next = [False for _ in range(target + 1)]
#     curr = [False for _ in range(target + 1)]
#     next[0] = True
#     curr[0] = True
#     for index in range(n - 1, -1, -1):
#         for target_sum in range(1, target + 1):
#             include = False
#             if target_sum - arr[index] >= 0:
#                 include = next[target_sum - arr[index]]
#             exclude = next[target_sum]
#             curr[target_sum] = include or exclude
#         next = curr[:]
#     return curr[target]
    
# def partition_equal_subset_sum(nums):
#     sum_of_elements = 0
#     for num in nums:
#         sum_of_elements += num
#     if sum_of_elements % 2 != 0:
#         return False
#     # ans = solve_rec(nums, 0, sum_of_elements // 2) # we change this problem into if target sum is present in array. 
#     # dp = [[-1 for _ in range((sum_of_elements//2)+ 1)]for _ in range(len(nums)+ 1)]
#     # ans = solve_mem(nums, 0, sum_of_elements // 2, dp)
#     # ans = solve_tab(nums, sum_of_elements // 2)
#     ans = solve_SO(nums, sum_of_elements // 2)
#     return ans
# print(partition_equal_subset_sum(nums))
 
# # #Minimum swaps to make subsequence increasing
# # arr1 = [1, 4, 3, 5]
# # arr2 = [1, 2, 5, 7]
# arr1 = [4, 3]
# arr2 = [2, 5]
# def solve_rec(arr1, arr2, index, swapped):
#     if index == len(arr1):
#         return 0
#     ans = float('inf')
#     prev1 = arr1[index - 1]
#     prev2 = arr2[index - 1]
#     if swapped:
#         prev1, prev2 = prev2, prev1
#     # no swap
#     if arr1[index] > prev1 and arr2[index] > prev2:
#         ans = solve_rec(arr1, arr2, index + 1, 0)
#     #swap
#     if arr1[index] > prev2 and arr2[index] > prev1:
#         ans = min(ans, 1 + solve_rec(arr1, arr2, index + 1, 1))
#     return ans

# def solve_mem(arr1, arr2, index, swapped, dp):
#     if index == len(arr1):
#         return 0
#     if dp[index][swapped] != -1:
#         return dp[index][swapped]
#     ans = float('inf')
#     prev1 = arr1[index - 1]
#     prev2 = arr2[index - 1]
#     if swapped:
#         prev1, prev2 = prev2, prev1
#     # no swap
#     if arr1[index] > prev1 and arr2[index] > prev2:
#         ans = solve_mem(arr1, arr2, index + 1, 0, dp)
#     #swap
#     if arr1[index] > prev2 and arr2[index] > prev1:
#         ans = min(ans, 1 + solve_mem(arr1, arr2, index + 1, 1, dp))
#     dp[index][swapped] = ans
#     return dp[index][swapped]

# def solve_tab(arr1, arr2):
#     n = len(arr1)
#     dp = [[0 for _ in range(2)]for _ in range(len(arr1) + 1)]

#     for index in range(n -1, -1, -1):
#         for swapped in range(1,-1,-1):
#             ans = float('inf')
#             prev1 = arr1[index - 1]
#             prev2 = arr2[index - 1]
#             if swapped:
#                 prev1, prev2 = prev2, prev1
#             # no swap
#             if arr1[index] > prev1 and arr2[index] > prev2:
#                 ans = dp[index + 1][0]
#             #swap
#             if arr1[index] > prev2 and arr2[index] > prev1:
#                 ans = min(ans, 1 + dp[index + 1][1])
#             dp[index][swapped] = ans
#     return dp[1][0]

# def solve_SO(arr1, arr2):
#     n = len(arr1)
#     # dp = [[0 for _ in range(2)]for _ in range(len(arr1) + 1)]
#     curr = [0 for _ in range(3)]
#     next = [0 for _ in range(3)]

#     for index in range(n - 1, 0, -1):
#         for swapped in range(1, -1, -1):
#             ans = float('inf')
#             prev1 = arr1[index - 1]
#             prev2 = arr2[index - 1]
            
#             if swapped:
#                 prev1, prev2 = prev2, prev1
#             # no swap
#             if arr1[index] > prev1 and arr2[index] > prev2:
#                 ans = next[0]
#             #swap
#             if arr1[index] > prev2 and arr2[index] > prev1:
#                 ans = min(ans, 1 + next[1])
#             curr[swapped] = ans
#         next = curr[:]
#     return next[0]


# def minimum_swaps(arr1, arr2):
#     arr1.insert(0, -1)
#     arr2.insert(0, -1)
#     # ans = solve_rec(arr1, arr2, 1, 0)
#     # dp = [[-1 for _ in range(2)]for _ in range(len(arr1) + 1)]
#     # ans = solve_mem(arr1, arr2, 1, 0, dp)
#     # ans = solve_tab(arr1, arr2)
#     ans = solve_SO(arr1, arr2)
#     return ans

# print(minimum_swaps(arr1, arr2))


# # Buy and sell stocks I
# # You are given an array prices where prices[i] is the price of a given stock on the ith day.
# # You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# # Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
# prices = [7,1,5,3,6,4]
# def solve_brute_force(prices):
#     max_profit = 0
#     n = len(prices)
#     for i in range(n - 1):
#         ans = 0
#         for j in range(i + 1, n):
#             curr_profit = prices[j] - prices[i]
#             ans = max(ans, curr_profit)
#         max_profit = max(max_profit, ans)
#     return max_profit
# def solve_optimised(prices):
#     mini = prices[0]
#     max_profit = 0
#     n = len(prices)
#     for i in range(1, n):
#         profit = prices[i] - mini
#         max_profit = max(profit, max_profit)
#         mini = min(mini, prices[i])
#     return max_profit
# def buy_and_sell_stock(prices):
#     # ans = solve_brute_force(prices)
#     ans = solve_optimised(prices)
#     return ans
# print(buy_and_sell_stock(prices))

# # Buy and Sell Stocks part II
# # You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
# # On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
# # Find and return the maximum profit you can achieve.
# prices = [7,1,5,3,6,4]
# def solve_rec(prices, index, buy):
#     if index == len(prices):
#         return 0
#     profit = 0
#     if buy:
#         buykro = -prices[index] + solve_rec(prices, index + 1, 0)
#         skipkro = 0 + solve_rec(prices, index + 1, 1)
#         profit  = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + solve_rec(prices, index + 1, 1)
#         skipkro = solve_rec(prices, index + 1, 0)
#         profit = max(sellkro, skipkro)
#     return profit

# def solve_mem(prices, index, buy, dp):
#     if index == len(prices):
#         return 0
#     if dp[index][buy] != -1:
#         return dp[index][buy]
#     profit = 0
#     if buy:
#         buykro = -prices[index] + solve_mem(prices, index + 1, 0, dp)
#         skipkro = 0 + solve_mem(prices, index + 1, 1, dp)
#         profit  = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + solve_mem(prices, index + 1, 1, dp)
#         skipkro = solve_mem(prices, index + 1, 0, dp)
#         profit = max(sellkro, skipkro)
#     dp[index][buy] = profit
#     return dp[index][buy]
# def solve_tab(prices):
#     n = len(prices)
#     dp = [[0 for _ in range(2)]for _ in range(n + 1)]

#     for index in range(n - 1, -1, -1):
#         for buy in range(1, -1, -1):
#             profit = 0
#             if buy:
#                 buykro = -prices[index] + dp[index + 1][0]
#                 skipkro = 0 + dp[index + 1][1]
#                 profit  = max(buykro, skipkro)
#             else:
#                 sellkro = prices[index] + dp[index + 1][1]
#                 skipkro = dp[index + 1][0]
#                 profit = max(sellkro, skipkro)
#             dp[index][buy] = profit
#     return dp[0][1]

# def solve_SO(prices):
#     n = len(prices)
#     # dp = [[0 for _ in range(2)]for _ in range(n + 1)]
#     curr = [0 for _ in range(2)]
#     next = [0 for _ in range(2)]

#     for index in range(n - 1, -1, -1):
#         for buy in range(1, -1, -1):
#             profit = 0
#             if buy:
#                 buykro = -prices[index] + next[0]
#                 skipkro = 0 + next[1]
#                 profit  = max(buykro, skipkro)
#             else:
#                 sellkro = prices[index] + next[1]
#                 skipkro = next[0]
#                 profit = max(sellkro, skipkro)
#             curr[buy] = profit
#             next = curr[:]
#     return next[1]

# def buy_and_sell_stocksII(prices):
#     # ans = solve_rec(prices, 0, 1)
#     # dp = [[-1 for _ in range(2)]for _ in range(len(prices) + 1)]
#     # ans = solve_mem(prices, 0, 1, dp)
#     # ans = solve_tab(prices)
#     # ans = solve_SO(prices)
#     # return ans
#     profit = 0
#     for i in range(len(prices) - 1):
#         if prices[i] < prices[i + 1]:
#             profit += prices[i + 1] - prices[i]
#     return profit
# print(buy_and_sell_stocksII(prices))


# # Buy And Sell Stocks III 
# # You are given an array prices where prices[i] is the price of a given stock on the ith day.
# # Find the maximum profit you can achieve. You may complete at most two transactions.
# # Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

# prices = [3,3,5,0,0,3,1,4]
# def solve_rec(prices, index, buy, limit):
#     if index == len(prices):
#         return 0
#     if limit == 0:
#         return 0
#     profit  = 0

#     if buy:
#         buykro = -prices[index] + solve_rec(prices, index + 1, 0, limit)
#         skipkro = solve_rec(prices, index + 1, 1, limit)
#         profit  = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + solve_rec(prices, index + 1, 1, limit - 1)
#         skipkro = solve_rec(prices, index + 1, 0, limit)
#         profit = max(sellkro, skipkro)
#     return profit

# def solve_mem(prices, index, buy, limit, dp):
#     if index == len(prices):
#         return 0
#     if limit == 0:
#         return 0
#     if dp[index][buy][limit] != -1:
#         return dp[index][buy][limit]
#     profit  = 0
#     if buy:
#         buykro = -prices[index] + solve_mem(prices, index + 1, 0, limit, dp)
#         skipkro = solve_mem(prices, index + 1, 1, limit, dp)
#         profit  = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + solve_mem(prices, index + 1, 1, limit - 1, dp)
#         skipkro = solve_mem(prices, index + 1, 0, limit, dp)
#         profit = max(sellkro, skipkro)
#     dp[index][buy][limit] = profit
#     return dp[index][buy][limit]
# def solve_tab(prices):
#     n = len(prices)
#     dp = [[[0 for _ in range(3) ]for _ in range(2)]for _ in range(len(prices) + 1)]
#     for index in range(n-1, -1, -1):
#         for buy in range(1, -1, -1):
#             for limit in range(1,3):
#                 profit  = 0
#                 if buy:
#                     buykro = -prices[index] + dp[index + 1][0][limit]
#                     skipkro = dp[index + 1][1][limit]
#                     profit  = max(buykro, skipkro)
#                 else:
#                     sellkro = prices[index] + dp[index + 1][1][limit - 1]
#                     skipkro = dp[index + 1][0][limit]
#                     profit = max(sellkro, skipkro)
#                 dp[index][buy][limit] = profit
#     return dp[0][1][2]

# def solve_SO(prices):
#     n = len(prices)
#     # dp = [[[0 for _ in range(3) ]for _ in range(2)]for _ in range(len(prices) + 1)]
#     curr =[[0 for _ in range(3) ]for _ in range(2)]
#     next =[[0 for _ in range(3) ]for _ in range(2)]
#     for index in range(n-1, -1, -1):
#         for buy in range(1, -1, -1):
#             for limit in range(1,3):
#                 profit  = 0
#                 if buy:
#                     buykro = -prices[index] + next[0][limit]
#                     skipkro = next[1][limit]
#                     profit  = max(buykro, skipkro)
#                 else:
#                     sellkro = prices[index] + next[1][limit - 1]
#                     skipkro = next[0][limit]
#                     profit = max(sellkro, skipkro)
#                 curr[buy][limit] = profit
#             curr = next[:]
#     return next[1][2]


# def buy_and_sell_stocksIII(prices):
#     # ans = solve_rec(prices, 0, 1, 2)
#     # dp = [[[-1 for _ in range(3) ]for _ in range(2)]for _ in range(len(prices) + 1)]
#     # ans = solve_mem(prices, 0, 1, 2, dp)
#     # ans = solve_tab(prices)
#     ans = solve_SO(prices)
#     return ans
# print(buy_and_sell_stocksIII(prices))

# # Buy and Sell Stock IV 
# # You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.
# # Find the maximum profit you can achieve. You may complete at most k transactions: i.e. you may buy at most k times and sell at most k times.
# # Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
# k = 2
# prices = [3,2,6,5,0,3]
# def solve_rec(prices, k, index, no_of_operations):
#     if index == len(prices):
#         return 0
#     if no_of_operations == 2 * k:
#         return 0
#     profit = 0
#     if no_of_operations & 1 != 1:
#         buykro = -prices[index] + solve_rec(prices, k, index + 1, no_of_operations + 1)
#         skipkro = solve_rec(prices, k, index + 1, no_of_operations)
#         profit = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + solve_rec(prices, k, index + 1, no_of_operations + 1)
#         skipkro = solve_rec(prices, k ,index + 1, no_of_operations)
#         profit = max(sellkro, skipkro)
#     return profit

# def solve_mem(prices, k, index, no_of_operations, dp):
#     if index == len(prices):
#         return 0
#     if no_of_operations == 2 * k:
#         return 0
#     if dp[index][no_of_operations] != -1:
#         return dp[index][no_of_operations]
#     profit = 0
#     if no_of_operations & 1 != 1:
#         buykro = -prices[index] + solve_mem(prices, k, index + 1, no_of_operations + 1, dp)
#         skipkro = solve_mem(prices, k, index + 1, no_of_operations, dp)
#         profit = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + solve_mem(prices, k, index + 1, no_of_operations + 1, dp)
#         skipkro = solve_mem(prices, k ,index + 1, no_of_operations, dp)
#         profit = max(sellkro, skipkro)
#     dp[index][no_of_operations] =  profit
#     return dp[index][no_of_operations]

# def solve_tab(prices, k):
#     n = len(prices)
#     dp = [[0 for _ in range((2 * k) + 1)]for _ in range(n + 1)]

#     for index in range(n - 1, -1, -1):
#         for no_of_operations in range(2 * k - 1, -1, -1):
#             profit = 0
#             if no_of_operations & 1 != 1:
#                 buykro = -prices[index] + dp[index + 1][no_of_operations + 1]
#                 skipkro = dp[index + 1][no_of_operations]
#                 profit = max(buykro, skipkro)
#             else:
#                 sellkro = prices[index] + dp[index + 1][no_of_operations + 1]
#                 skipkro = dp[index + 1][no_of_operations]
#                 profit = max(sellkro, skipkro)
#             dp[index][no_of_operations] =  profit
#     return dp[0][0]

# def solve_SO(prices, k):
#     n = len(prices)
#     # dp = [[0 for _ in range((2 * k) + 1)]for _ in range(n + 1)]
#     curr = [0 for _ in range((2 * k) + 1)]
#     next = [0 for _ in range((2 * k) + 1)]

#     for index in range(n - 1, -1, -1):
#         for no_of_operations in range(2 * k - 1, -1, -1):
#             profit = 0
#             if no_of_operations & 1 != 1:
#                 buykro = -prices[index] + next[no_of_operations + 1]
#                 skipkro = next[no_of_operations]
#                 profit = max(buykro, skipkro)
#             else:
#                 sellkro = prices[index] + next[no_of_operations + 1]
#                 skipkro = next[no_of_operations]
#                 profit = max(sellkro, skipkro)
#             curr[no_of_operations] =  profit
#         next = curr[:]
#     return next[0]


        
# def buy_and_sell_stocks_IV(prices, k):
#     # ans = solve_rec(prices, k, 0, 0)
#     # dp = [[-1 for _ in range((2 * k) + 1)]for _ in range(len(prices) + 1)]
#     # ans = solve_mem(prices, k, 0, 0, dp)
#     # ans = solve_tab(prices, k)
#     ans = solve_SO(prices, k)
#     return ans
# print(buy_and_sell_stocks_IV(prices, k))


# # Best Time to Buy and Sell Stock with Transaction Fee
# # You are given an array prices where prices[i] is the price of a given stock on the ith day, and an integer fee representing a transaction fee.
# # Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.
# # Note:
# # You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
# # The transaction fee is only charged once for each stock purchase and sale.

# prices = [1,3,2,8,4,9]
# fee = 2
# def solve_rec(prices, index, buy, fee):
#     if index == len(prices):
#         return 0
#     profit = 0
#     if buy:
#         buykro = -prices[index] + solve_rec(prices, index + 1, 0, fee)
#         skipkro = 0 + solve_rec(prices, index + 1, 1, fee)
#         profit  = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + (-fee) + solve_rec(prices, index + 1, 1, fee)
#         skipkro = solve_rec(prices, index + 1, 0, fee)
#         profit = max(sellkro, skipkro)
#     return profit
# def solve_mem(prices, index, buy, fee, dp):
#     if index == len(prices):
#         return 0
#     if dp[index][buy] != -1:
#         return dp[index][buy]
#     profit = 0
#     if buy:
#         buykro = -prices[index] + solve_mem(prices, index + 1, 0, fee, dp)
#         skipkro = 0 + solve_mem(prices, index + 1, 1, fee, dp)
#         profit  = max(buykro, skipkro)
#     else:
#         sellkro = prices[index] + (-fee) + solve_mem(prices, index + 1, 1, fee, dp)
#         skipkro = solve_mem(prices, index + 1, 0, fee, dp)
#         profit = max(sellkro, skipkro)
#     dp[index][buy] = profit
#     return dp[index][buy]
# def solve_tab(prices, fee):
#     dp = [[0 for _ in range(2)]for _ in range(len(prices) + 1)]
#     n = len(prices)
#     for index in range(n - 1, -1, -1):
#         for buy in range(1, -1, -1):
#             profit = 0
#             if buy:
#                 buykro = -prices[index] + dp[index + 1][0]
#                 skipkro = 0 + dp[index + 1][1]
#                 profit  = max(buykro, skipkro)
#             else:
#                 sellkro = prices[index] - fee + dp[index + 1][1]
#                 skipkro = dp[index + 1][0]
#                 profit = max(sellkro, skipkro)
#             dp[index][buy] = profit
#     return dp[0][1]

# def solve_SO(prices, fee):
#     n = len(prices)
#     # dp = [[0 for _ in range(2)]for _ in range(n + 1)]
#     curr = [0 for _ in range(2)]
#     next = [0 for _ in range(2)]

#     for index in range(n - 1, -1, -1):
#         for buy in range(1, -1, -1):
#             profit = 0
#             if buy:
#                 buykro = -prices[index] + next[0]
#                 skipkro = 0 + next[1]
#                 profit  = max(buykro, skipkro)
#             else:
#                 sellkro = prices[index] - fee + next[1]
#                 skipkro = next[0]
#                 profit = max(sellkro, skipkro)
#             curr[buy] = profit
#             next = curr[:]
#     return next[1]

# def buy_and_sell_stocks_with_transaction_fees(prices, fee):
#     # ans = solve_rec(prices, 0, 1, fee)
#     # dp = [[-1 for _ in range(2)]for _ in range(len(prices) + 1)]
#     # ans = solve_mem(prices, 0, 1 ,fee, dp)
#     # ans = solve_tab(prices, fee)
#     ans = solve_SO(prices, fee)
#     return ans
# print(buy_and_sell_stocks_with_transaction_fees(prices, fee))

# # Longest Arithmetic Subsequence
# arr = [1, 7, 9, 10, 13, 14, 19, 25]
# def solve_rec(arr, index, diff):
#     if index < 0:
#         return 0
#     ans = 0
#     for k in range(index - 1, -1, -1):
#         if arr[index] - arr[k] == diff:
#             ans = max(ans, 1 + solve_rec(arr, k, diff))
#     return ans
# def solve_mem(arr, index, diff, dp):
#     if index < 0:
#         return 0
#     if (index, diff) in dp:
#         return dp[(index, diff)]
#     ans = 0
#     for k in range(index - 1, -1, -1):
#         if arr[index] - arr[k] == diff:
#             ans = max(ans, 1 + solve_mem(arr, k, diff, dp))
#     dp[(index, diff)] = ans
#     return ans
# def solve_tab(arr):
#     n = len(arr)
#     dp = {}
#     ans = 0
#     for i in range(1, n):
#         for j in range(i):
#             count = 1
#             diff = arr[i] - arr[j]
#             if (j, diff) in dp:
#                 count = dp[(j, diff)]
#             dp[(i, diff)] = 1 + count
#             ans = max(ans, dp[(i, diff)])
#     return ans
            
            

# def longest_arithmetic_subsequence(arr):
#     # n = len(arr)
#     # if n <= 2:
#     #     return n
#     # ans = 0
#     # for i in range(n - 1):
#     #     for j in range(i + 1, n):
#     #         diff = arr[j] - arr[i]
#     #         ans = max(ans, 2 + solve_rec(arr, i, diff))
#     # return ans

#     # n = len(arr)
#     # if n <= 2:
#     #     return n
#     # ans = 0
#     # dp = {}
#     # for i in range(n - 1):
#     #     for j in range(i + 1, n):
#     #         diff = arr[j] - arr[i]
#     #         ans = max(ans, 2 + solve_mem(arr, i, diff, dp))
#     # return ans

#     n = len(arr)
#     if n <= 2:
#         return n
#     ans = solve_tab(arr)
#     return ans
# print(longest_arithmetic_subsequence(arr))

# # Longest Arithmetic Subsequence with common difference d
# arr = [1, 7, 9, 10, 13, 14, 19, 25]
# common_diff = 3
# def longest_arithmetic_subsequence_with_diff(arr, diff):
#     ans = 0
#     dp = {}
#     for i in range(len(arr)):
#         if  (arr[i] - diff) in dp:
#             dp[arr[i]] = dp[arr[i] - diff] + 1
#         else:
#             dp[arr[i]]  = 1
#         ans = max(ans, dp[arr[i]])
#     return ans

# print(longest_arithmetic_subsequence_with_diff(arr, common_diff))

# # catlan number or unique binary search tree
# # Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.
# hint  = f(i) = summation of f(i - 1) * f(n - i)
# n = 5
# def solve_rec(n):
#     if n <=1 :
#         return 1
#     ans = 0
#     for i in range(1, n + 1):
#         ans += solve_rec(i - 1) * solve_rec(n - i)
#     return ans
# def solve_mem(n, dp):
#     if n <=1 :
#         return 1
#     if dp[n] != -1:
#         return dp[n]
#     ans = 0
#     for i in range(1, n + 1):
#         ans += solve_mem(i - 1, dp) * solve_mem(n - i, dp)
#     dp[n] = ans
#     return ans
# def solve_tab(n):
#     dp = [0 for _ in range(n + 2)]
#     dp[0] = 1
#     dp[1] = 1
#     for j in range(2, n + 1):
#         ans = 0
#         for i in range(1, j + 1):
#             ans += dp[i - 1] * dp[j - i]
#         dp[j] = ans
#     return dp[n]

# def no_of_unique_binary_tree(n):
#     # ans = solve_rec(n)
#     # dp = [-1 for _ in range(n + 2)]
#     # ans = solve_mem(n, dp)
#     ans = solve_tab(n)
#     return ans
# print(no_of_unique_binary_tree(n))

# # Guess number higher or lower II
# # We are playing the Guessing Game. The game will work as follows:
# # I pick a number between 1 and n.
# # You guess a number.
# # If you guess the right number, you win the game.
# # If you guess the wrong number, then I will tell you whether the number I picked is higher or lower, and you will continue guessing.
# # Every time you guess a wrong number x, you will pay x dollars. If you run out of money, you lose the game.
# # Given a particular n, return the minimum amount of money you need to guarantee a win regardless of what number I pick.

# n = 10
# def solve_rec(start, end):
#     if start >= end:
#         return 0
#     cost = float('inf')
#     for i in range(start, end + 1):
#         cost = min(cost, i + max(solve_rec(start, i - 1), solve_rec(i + 1, end)))
#     return cost
# def solve_mem(start, end, dp):
#     if start >= end:
#         return 0
#     if (start, end) in dp:
#         return dp[(start, end)]
#     cost = float('inf')
#     for i in range(start, end + 1):
#         cost = min(cost, i + max(solve_mem(start, i - 1, dp), solve_mem(i + 1, end, dp)))
#     dp[(start, end)] = cost
#     return cost
# def solve_tab(n):
#     dp = [[0 for _ in range(n + 2)] for _ in range(n + 2)]
#     for start in range(n, 0, -1):
#         for end in range(start, n + 1, 1):
#             if start == end:# this is the case when we guess the number correct
#                 continue
#             else:
#                 cost = float('inf')
#                 for i in range(start, end + 1):
#                     cost = min(cost, i + max(dp[start][i - 1], dp[i + 1][end]))
#                 dp[start][end] = cost
#     return dp[1][n]
            

# def number_higher_or_lowerII(n):
#     # ans = solve_rec(1, n)
#     # dp = {}
#     # ans = solve_mem(1, n, dp)
#     ans = solve_tab(n)
#     return ans
# print(number_higher_or_lowerII(n))

# # Minimum cost tree from leaf values
# # Given an array arr of positive integers, consider all binary trees such that:
# # Each node has either 0 or 2 children;
# # The values of arr correspond to the values of each leaf in an in-order traversal of the tree.
# # The value of each non-leaf node is equal to the product of the largest leaf value in its left and right subtree, respectively.
# # Among all possible binary trees considered, return the smallest possible sum of the values of each non-leaf node. It is guaranteed this sum fits into a 32-bit integer.
# # A node is a leaf if and only if it has zero children.

# arr = [6, 2, 4, 5]
# def solve_rec(arr, map, left, right):
#     if left == right :
#         return 0
#     ans = float('inf')
#     for i in range(left, right):
#         ans = min(ans, map[(left, i)] * map[(i + 1, right)] + solve_rec(arr, map, left, i) + solve_rec(arr, map, i + 1, right))
#     return ans

# def solve_mem(arr, map, left, right, dp):
#     if left == right :
#         return 0
#     if (left, right) in dp:
#         return dp[(left, right)]
#     ans = float('inf')
#     for i in range(left, right):
#         ans = min(ans, map[(left, i)] * map[(i + 1, right)] + solve_mem(arr, map, left, i, dp) + solve_mem(arr, map, i + 1, right, dp))
#     dp[(left,right)] = ans
#     return ans
    
# def solve_tab(arr, map):
#     n  = len(arr)
#     dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
#     for left in range(n - 1, -1, -1):
#         for right in range(left + 1, n):
#             ans = float('inf')
#             for i in range(left, right):
#                 ans = min(ans, map[(left, i)] * map[(i + 1, right)] + dp[left][i] + dp[i + 1][right])
#             dp[left][right] = ans
#     return dp[0][n - 1]

# def minimum_cost_tree_leaf_values(arr):
#     n = len(arr)
#     map = {} # stores maximum for each range
#     for i in range(n):
#         map[(i, i)] = arr[i]
#         for j in range(i + 1, n):
#             map[(i, j)] = max(arr[j], map[(i, j - 1)])

#     # ans = solve_rec(arr, map, 0, n - 1)
#     # dp = {}
#     # ans = solve_mem(arr, map, 0, n - 1, dp)
#     ans = solve_tab(arr, map)
#     return ans
# print(minimum_cost_tree_leaf_values(arr))

# # Longest Common Subsequence 
# str1 = 'abcdeg'
# str2 = 'acefsdg'
# def solve_rec(str1, str2, i, j):
#     if i == len(str1):
#         return 0
#     if j == len(str2):
#         return 0
#     ans = 0
#     if str1[i] ==str2[j]:
#         ans = 1 + solve_rec(str1, str2, i + 1, j + 1)
#     else:
#         ans = max(solve_rec(str1, str2, i + 1, j), solve_rec(str1, str2, i, j + 1))
#     return ans
# def solve_mem(str1, str2, i, j, dp):
#     if i == len(str1):
#         return 0
#     if j == len(str2):
#         return 0
#     if dp[i][j] != -1:
#         return dp[i][j]
#     ans = 0
#     if str1[i] ==str2[j]:
#         ans = 1 + solve_mem(str1, str2, i + 1, j + 1, dp)
#     else:
#         ans = max(solve_mem(str1, str2, i + 1, j, dp), solve_mem(str1, str2, i, j + 1, dp))
#     dp[i][j] = ans
#     return ans
# def solve_tab(str1, str2):
#     dp =  [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
#     for i in range(len(str1) - 1, -1, -1):
#         for j in range(len(str2) - 1, -1, -1):
#             if str1[i] == str2[j]:
#                 ans = 1 + dp[i + 1][j + 1]
#             else:
#                 ans = max(dp[i + 1][j],dp[i][j + 1])
#             dp[i][j] = ans
#     return dp[0][0]

# def solve_SO(str1, str2):
#     # dp =  [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
#     next = [0 for _ in range(len(str2) + 1)]
#     curr = [0 for _ in range(len(str2) + 1)]
#     for i in range(len(str1) - 1, -1, -1):
#         for j in range(len(str2) - 1, -1, -1):
#             if str1[i] == str2[j]:
#                 ans = 1 + next[j + 1]
#             else:
#                 ans = max(next[j],curr[j + 1])
#             curr[j] = ans
#         next = curr[:]
#     return next[0]



# def longest_common_subsequence(str1, str2):
#     # ans = solve_rec(str1, str2, 0, 0)
#     # dp = [[-1 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
#     # ans = solve_mem(str1, str2, 0, 0, dp)
#     # ans = solve_tab(str1, str2)
#     ans = solve_SO(str1, str2)
#     return ans
# print(longest_common_subsequence(str1, str2))

# # Longest Palindromic Subsequence
# # Hint-  rev the string and then find longest common subsequence
# str = 'bbabbb'
# def solve_longest_common_subsequence(str1, str2):
#     next = [0 for _ in range(len(str1) + 1)]
#     curr = [0 for _ in range(len(str1) + 1)]
#     for i in range(len(str1) - 1, -1, -1):
#         for j in range(len(str2) - 1, -1, -1):
#             ans = 0
#             if str1[i] == str2[j]:
#                 ans = 1 + next[j + 1]
#             else:
#                 ans = max(next[j], curr[j + 1])
#             curr[j] = ans
#         next = curr[:]
#     return next[0]

# def longest_palindromic_subsequence(str):
#     rev_str = str[::-1]
#     ans = solve_longest_common_subsequence(str, rev_str)
#     return ans
# print(longest_palindromic_subsequence(str))

# # Maximal Rectangle
# matrix = [["1","0","1","0","0"],
#           ["1","0","1","1","1"],
#           ["1","1","1","1","1"],
#           ["1","0","0","1","0"]
#           ]
# def next_smaller_element_index(arr):
#     n = len(arr)
#     ans_arr = [-1] * n
#     stack =[]
#     for i in range(n-1, -1, -1):
#         while stack and arr[stack[-1]] >= arr[i]:
#             stack.pop()
#         if stack:
#             ans_arr[i] = stack[-1]
#         stack.append(i)
#     return ans_arr

# def prev_smaller_element_index(arr):
#     n = len(arr)
#     ans_arr = [-1] * n
#     stack =[]
#     for i in range(n):
#         while stack and arr[stack[-1]] >= arr[i]:
#             stack.pop()
#         if stack:
#             ans_arr[i] = stack[-1]
#         stack.append(i)
#     return ans_arr

# def find_histrogram_area(arr):
#     n = len(arr)
#     next = next_smaller_element_index(arr)
#     prev = prev_smaller_element_index(arr)
#     max_area = 0
#     for i in range(n):
#         area = 0
#         length = arr[i]
#         if next[i] == -1:
#             next[i] = n
#         width = next[i] - prev[i] - 1
#         area = length * width
#         max_area = max(area ,max_area)
#     return max_area

# def maximal_rectangle(matrix):
#     max_area = 0
#     n = len(matrix)
#     m = len(matrix[0])
#     histogram = [0] * m
#     for i in range(n):
#         for j in range(m):
#             if matrix[i][j] == '1':
#                 histogram[j] += 1
#             else:
#                 histogram[j] = 0
#         area = find_histrogram_area(histogram)
#         max_area = max(max_area, area)
#     return max_area
# print(maximal_rectangle(matrix))



# # Wildcard Pattern
# str = 'abcde'
# pattern = 'ab*c?e'
# def solveRec(str, pattern, i, j):
#     if i == -1 and j == -1:
#         return True
#     if i >= 0 and j == -1:
#         return False
#     if i == -1 and j >= 0:
#         for k in range(j):
#             if pattern[k] != '*':
#                 return False
#         return True
#     if str[i] == pattern[j] or pattern[j] == '?':
#         return solveRec(str, pattern, i - 1, j - 1)
#     elif pattern[j] == '*':
#         return solveRec(str, pattern, i - 1, j) or solveRec(str, pattern, i, j - 1)
#     else :
#         return False

# def solveMem(str, pattern, i, j, dp):
#     if i == -1 and j == -1:
#         return True
#     if i >= 0 and j == -1:
#         return False
#     if i == -1 and j >= 0:
#         for k in range(j):
#             if pattern[k] != '*':
#                 return False
#         return True
#     if (i, j) in dp:
#         return dp[(i, j)]
#     if str[i] == pattern[j] or pattern[j] == '?':
#         dp[(i, j)] = solveMem(str, pattern, i - 1, j - 1, dp)
#     elif pattern[j] == '*':
#         dp[(i, j)] = (solveMem(str, pattern, i - 1, j, dp) or solveMem(str, pattern, i, j - 1, dp))
#     else:
#         dp[(i, j)] = False
#     return dp[(i, j)]

# def solveMem1(str, pattern, i, j, dp):
#     if i == 0 and j == 0:
#         return True
#     if i > 0 and j == 0:
#         return False
#     if i == 0 and j > 0:
#         for k in range(1, j + 1):
#             if pattern[k - 1] != '*':
#                 return False
#         return True
#     if  dp[i][j] != -1:
#         return dp[i][j]
#     if str[i - 1] == pattern[j - 1] or pattern[j - 1] == '?':
#         dp[i][j] = solveMem1(str, pattern, i - 1, j - 1, dp)
#     elif pattern[j - 1] == '*':
#         dp[i][j] = (solveMem1(str, pattern, i - 1, j, dp) or solveMem1(str, pattern, i, j - 1, dp))
#     else:
#         dp[i][j] = False
#     return dp[i][j]

# def solveTab(str, pattern):
#     dp = [[0 for _ in range(len(pattern) + 1)]for _ in range(len(str) + 1)]
#     dp[0][0] = True
#     for j in range(1, len(pattern) + 1):
#         bool = True
#         for k in range(1, j + 1):
#             if pattern[k - 1] != '*':
#                 bool = False
#                 break
#         dp[0][j] = bool
    
#     for i in range(1, len(str) + 1):
#         for j in range(1, len(pattern) + 1):
#             if str[i - 1] == pattern[j - 1] or pattern[j - 1] == '?':
#                 dp[i][j] = dp[i - 1][j - 1]
#             elif pattern[j - 1] == '*':
#                 dp[i][j] = (dp[i - 1][j] or dp[i][j - 1])
#             else:
#                 dp[i][j] = False
#     return dp[len(str)][len(pattern)]

# def solveSO(str, pattern):
#     # dp = [[0 for _ in range(len(pattern) + 1)]for _ in range(len(str) + 1)]
#     prev = [0 for _ in range(len(pattern) + 1)]
#     curr = [0 for _ in range(len(pattern) + 1)]

#     prev[0] = True
#     for j in range(1, len(pattern) + 1):
#         bool = True
#         for k in range(1, j + 1):
#             if pattern[k - 1] != '*':
#                 bool = False
#                 break
#         prev[j] = bool
    
#     for i in range(1, len(str) + 1):
#         for j in range(1, len(pattern) + 1):
#             if str[i - 1] == pattern[j - 1] or pattern[j - 1] == '?':
#                 curr[j] = prev[j - 1]
#             elif pattern[j - 1] == '*':
#                 curr[j] = (prev[j] or curr[j - 1])
#             else:
#                 curr[j] = False
#         prev = curr[:]
#     return prev[len(pattern)]
    
# def wildcard_pattern(str, pattern):
#     # dp = {}
#     # ans = solveRec(str, pattern, len(str) - 1, len(pattern) - 1)
#     # ans = solveMem(str, pattern, len(str) - 1, len(pattern) - 1, dp)
#     # 1 based indexing 
#     # dp = [[-1 for _ in range(len(pattern) + 1)]for _ in range(len(str) + 1)]
#     # ans = solveMem1(str, pattern, len(str), len(pattern), dp)
#     # ans = solveTab(str, pattern)
#     ans = solveSO(str, pattern)
#     return ans

# print(wildcard_pattern(str, pattern))

# BACKTRACKING 
# # a rat in a maze
# maze = [
#     [1, 0, 0, 0],
#     [1, 1, 0, 1],
#     [1, 1, 0, 0],
#     [0, 1, 1, 1]
# ]
# def isSafe(newi, newj, visited, n, matrix):
#     if (newi>= 0 and newi < n) and (newj>= 0 and newj < n) and (matrix[newi][newj] == 1) and (visited[newi][newj] != 1):
#         return True
#     return False
# def solveRec(i, j, matrix, ans, path, visited, n):
#     if i == n-1 and j == n - 1:
#         ans.append(path)
#         return
#     visited[i][j] = 1
#     #Down
#     if isSafe(i + 1, j, visited, n, matrix):
#         solveRec(i + 1, j, matrix, ans, path + 'D', visited, n)
#     #Left
#     if isSafe(i, j - 1, visited, n, matrix):
#         solveRec(i, j - 1, matrix, ans, path + 'L', visited, n)
        
#     #Right
#     if isSafe(i, j + 1, visited, n, matrix):
#         solveRec(i, j + 1, matrix, ans, path + 'R', visited, n)
#     #Up
#     if isSafe(i - 1, j, visited, n, matrix):
#         solveRec(i - 1, j, matrix, ans, path + 'U', visited, n)
        
#     visited[i][j] = 0

# def rat_in_a_maze(maze):
#     ans = []
#     n = len(maze)
#     visited = [[0  for _ in range(n)]for _ in range(n)]
#     path = ''
#     if maze[0][0] == 1:
#         solveRec(0, 0, maze, ans, path, visited, n)
#     return ans
# ans = rat_in_a_maze(maze)
# print(ans)


# # N Queens Problem
# n = 5
# def add_matrix_to_ans(matrix, ans, n):
#     temp = []
#     for i in range(n):
#         row = []
#         for j in range(n):
#             row.append(matrix[i][j])
#         temp.append(row)
#     for row in temp:
#         print(row)
#     print('Next cinfiguration')
#     ans.append(temp)
# def isSafe(row, col, matrix, n):
#     x= row
#     y = col

#     #left check 
#     while y >= 0:
#         if matrix[x][y] == 1:
#             return False
#         y -=1
#     #lower Diaginal Check
#     x = row
#     y = col

#     while x < n and y >=0:
#         if matrix[x][y] == 1:
#             return False
#         x += 1
#         y -= 1
#     # Upper Diagonal Check
#     x = row
#     y = col
#     while x >= 0 and y >=0 :
#         if matrix[x][y] == 1:
#             return False
#         x -= 1
#         y -= 1
#     return True
    
# def solveRec(col, matrix, combined_ans, n):
#     if col == n:
#         add_matrix_to_ans(matrix, combined_ans, n)
#         return
#     for row in range(n):
#         if isSafe(row, col, matrix, n):
#             matrix[row][col] = 1
#             solveRec(col + 1, matrix, combined_ans, n)
#             matrix[row][col] = 0
    
# def n_queens_problem(n):
#     matrix = [[0 for _ in range(n)]for _ in range(n)]
#     combined_ans = []
#     solveRec(0, matrix, combined_ans, n)
#     return combined_ans
# ans = n_queens_problem(n)
# # print(ans)

# # Sudoko Solver
# board = [   ["5","3",".",".","7",".",".",".","."],
#             ["6",".",".","1","9","5",".",".","."],
#             [".","9","8",".",".",".",".","6","."],
#             ["8",".",".",".","6",".",".",".","3"],
#             ["4",".",".","8",".","3",".",".","1"],
#             ["7",".",".",".","2",".",".",".","6"],
#             [".","6",".",".",".",".","2","8","."],
#             [".",".",".","4","1","9",".",".","5"],
#             [".",".",".",".","8",".",".","7","9"]
#         ]
# def isSafe(row, col, board, value, n): 
#     for i in range(n):
#         #check value present in row
#         if board[row][i] == f'{value}':
#             return False
#         #check value present in col
#         if board[i][col] == f'{value}':
#             return False
#         #check value present in 3 X 3 matrix
#         if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == f'{value}':
#             return False
#     return True
        
# def solveRec(board, n):
#     for row in range(n):
#         for col in range(n):
#             if board[row][col] == '.':
#                 for value in range(1, n + 1 , 1):
#                     if isSafe(row, col, board, value, n):
#                         board[row][col] = f'{value}'
#                         solutionPossible = solveRec(board, n)
#                         if solutionPossible:
#                             return True
#                         else:
#                             # backtrack
#                             board[row][col] = '.'
#                 return False      
#     return True

# def sudoko_solver(board):
#     n = len(board)
#     # solveRec(board, n)
#     if solveRec(board, n):
#         for row in board:
#             print(row)
#     else:
#         print("No solution exists")
#     return
# sudoko_solver(board)

# # Tawor of Hanoi
# n = 3
# def solveRec(n, src, helper, dest):
#     if n == 1:
#         print(f'{src} + {dest}')
#         print(f'Transfer Disk {n} from {src} to {dest}')
#         return
#     solveRec(n-1, src, dest, helper)
#     print(f'Transfer Disk {n} from {src} to {dest}')
#     solveRec(n-1, helper, src, dest)
# def tower_of_hanoi(n):
#     solveRec(n, 'Src', 'Helper', 'Dest')
#     return
# tower_of_hanoi(n)
        
# # Unique path 2 print all paths
# obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
# def isSafe(newi, newj, visited, matrix, m, n):
#     if newi < m and newj < n and matrix[newi][newj] != 1 and visited[newi][newj] != 0:
#         return True
#     return False
# def solveRec(i, j, m, n, visited, matrix, path):
#     if i>= m or j >= n:
#         return
    
#     if i == m-1 and j == n-1:
#         print(path)
#         return 
#     visited[i][j] = 0
#     # down
#     if isSafe(i + 1, j ,visited, matrix, m, n):
#         solveRec(i + 1, j, m, n, visited, matrix, path + 'D')
#     # right
#     if isSafe(i, j + 1, visited, matrix, m, n):
#         solveRec(i, j + 1, m, n, visited, matrix, path + "R")

#     visited[i][j] = -1
#     return
    
    
# def unique_path_2(matrix):
#     visited = [[-1 for _ in range(len(matrix[0]))]for _ in range(len(matrix))]
#     m = len(matrix)
#     n = len(matrix[0])
#     if matrix[0][0] == 1 or matrix[m-1][n-1] == 1:
#         return 0
#     path = ''
#     ans = solveRec(0, 0, len(matrix), len(matrix[0]), visited, matrix, path)
#     return ans 
# unique_path_2(obstacleGrid)


#Unique path 2 count only no of ways
# def solveRec(i, j, m, n, matrix):
#     # If out of bounds or on an obstacle, return 0
#     if i >= m or j >= n or matrix[i][j] == 1:
#         return 0
    
#     # If we have reached the bottom-right corner, return 1
#     if i == m-1 and j == n-1:
#         return 1

#     # Recursively calculate paths moving down and right
#     downpaths = solveRec(i + 1, j, m, n, matrix)
#     rightpaths = solveRec(i, j + 1, m, n, matrix)

#     return downpaths + rightpaths

# def solveMem(i, j, m, n, matrix, dp):
#     # If out of bounds or on an obstacle, return 0
#     if i >= m or j >= n or matrix[i][j] == 1:
#         return 0
    
#     # If we have reached the bottom-right corner, return 1
#     if i == m-1 and j == n-1:
#         return 1
#     if (i, j) in dp:
#         return dp[(i, j)]
#     # Recursively calculate paths moving down and right
#     downpaths = solveMem(i + 1, j, m, n, matrix, dp)
#     rightpaths = solveMem(i, j + 1, m, n, matrix, dp)

#     dp[(i, j)] = downpaths + rightpaths
#     return dp[(i, j)]

# def solveTab(m, n, obstacleGrid):
#     dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
#     dp[m-1][n-1] = 1

#     for i in range(m-1, -1 ,-1):
#         for j in range(n-1, -1, -1):
#             if obstacleGrid[i][j] == 1:
#                 dp[i][j]  = 0
#             else:
#                 downpaths = dp[i + 1][j]
#                 rightpaths = dp[i][j + 1]
#                 dp[i][j] += downpaths + rightpaths
#     return dp[0][0]

# def solveSO(m, n, obstacleGrid):
#     # dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
#     curr = [0 for _ in range(n + 1)]
#     next = [0 for _ in range(n + 1)]
#     next[n-1] = 1

#     for i in range(m-1, -1 ,-1):
#         for j in range(n-1, -1, -1):
#             if obstacleGrid[i][j] == 1:
#                 curr[j]  = 0
#             else:
#                 downpaths = next[j]
#                 rightpaths = curr[j + 1]
#                 curr[j] = downpaths + rightpaths
#         next = curr[:]
#     return curr[0]

# def unique_path_2(matrix):
#     m = len(matrix)
#     n = len(matrix[0])
#     if matrix[0][0] == 1 or matrix[m-1][n-1] == 1:
#         return 0
#     # dp = {}
#     # return solveMem(0, 0, m, n, matrix, dp)
#     # return solveTab(m, n, matrix)
#     return solveSO(m, n, matrix)
#     # return solveRec(0, 0, m, n, matrix)

# obstacleGrid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
# ans = unique_path_2(obstacleGrid)
# print(ans)  


# # Find minimum cost
# days = [1, 5, 11, 30]
# cost = [3, 5, 8]
# def solveRec(cost, days, index, n):
#     if index >= n:
#         return 0
#     days1cost = cost[0] + solveRec(cost, days, index + 1, n)

#     i = index 
#     while i < n and days[i] < days[index] + 7:
#         i += 1
#     days7cost = cost[1] + solveRec(cost, days, i, n)
#     i = index
#     while i < n and days[i] < days[index] + 30:
#         i +=1
#     days30cost = cost[2] + solveRec(cost, days, i, n)
#     return min(days1cost, days7cost, days30cost)

# def solveMem(cost, days, index, n, dp):
#     if index >= n:
#         return 0
#     if dp[index] != -1:
#         return dp[index]
#     days1cost = cost[0] + solveMem(cost, days, index + 1, n, dp)

#     i = index 
#     while i < n and days[i] < days[index] + 7:
#         i += 1
#     days7cost = cost[1] + solveMem(cost, days, i, n, dp)
#     i = index
#     while i < n and days[i] < days[index] + 30:
#         i +=1
#     days30cost = cost[2] + solveMem(cost, days, i, n, dp)
#     dp[index] = min(days1cost, days7cost, days30cost)
#     return dp[index]

# def solveTab(cost, days, n):
#     dp = [float('inf') for _ in range(n + 1)]
#     dp[n] = 0
#     for index in range(n-1, -1, -1):
#         days1cost = cost[0] + dp[index + 1]

#         i = index 
#         while i < n and days[i] < days[index] + 7:
#             i += 1
#         days7cost = cost[1] + dp[i]
#         i = index
#         while i < n and days[i] < days[index] + 30:
#             i +=1
#         days30cost = cost[2] + dp[i]
#         dp[index] = min(days1cost, days7cost, days30cost)
#     return dp[0]

# def minimum_cost(cost, days):
#     n = len(days)
#     # ans = solveRec(cost, days, 0, n)
#     dp = [-1 for _ in range(n + 1)]
#     # ans = solveMem(cost, days, 0, n, dp)
#     ans = solveTab(cost, days, n)
#     return  ans
# print(minimum_cost(cost, days))


# def findTriplets(arr):
#     # Your code here
#     index_dict = {}
#     for i, num in enumerate(arr):
#         if num not in index_dict:
#             index_dict[num] = []
#         index_dict[num].append(i)
#     print(index_dict)

#     # Sort the array and store it in a new list
#     sorted_arr = sorted(arr)
#     n = len(sorted_arr)
#     triplets = []

#     for i in range(n - 2):
#         left = i + 1
#         right = n - 1
#         while left < right:
#             curr_sum = sorted_arr[i] + sorted_arr[left] + sorted_arr[right]
#             if curr_sum == 0:
#                 triplet = [index_dict[sorted_arr[i]].pop(0), 
#                             index_dict[sorted_arr[left]].pop(0), 
#                             index_dict[sorted_arr[right]].pop(0)]
#                 print(triplet)
                
#                 # Sort indices to maintain ascending order
#                 triplets.append(sorted(triplet))
                
#                 # Move pointers and skip duplicates
#                 left += 1
#                 right -= 1
#                 while left < right and sorted_arr[left] == sorted_arr[left - 1]:
#                     left += 1
#                 while left < right and sorted_arr[right] == sorted_arr[right + 1]:
#                     right -= 1

#             elif curr_sum < 0:
#                 left += 1
#             else:
#                 right -= 1

#     return triplets
# arr = [0, -1, 2, -3, 1]
# findTriplets(arr)
