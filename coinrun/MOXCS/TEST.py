def minimum(a, n):
    # inbuilt function to find the position of minimum
    minpos = a.index(min(a))

    # inbuilt function to find the position of maximum
    maxpos = a.index(max(a))

    # printing the position
    print("The maximum is at position", maxpos)
    print("The minimum is at position", minpos)


# driver code
a = [3, 4, 1, 3, 4, 5]
minimum(a, len(a))