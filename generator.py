import itertools
import os.path

from math import floor, ceil

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)

POSSIBLE_WORDS_LIST_FILE = os.path.join(DATA_DIR, "possible_words.txt")


def generate_3d_result_solutions():
    """
    Since nerdle has 8 digit solutions for results to be 3 digit numbers we have
    some different open formulas:

    aa+b=ccc || a+bb=ccc || a*bb=ccc || aa*b=ccc

    Let 'a','b' ∈ N where at least one, a or b must have two digits and the other must have one.
    Since 'a' is lesser than 100 and 'b' lesser than 10 but c greater than 99 then

    Sum: a+b=c
        This gives us the next domain: a ∈ [91;99] && b ∈ [1;9] => c ∈ [100;109]

    Multiplication: a*b=c
        This gives us the next domain: a ∈ [50;99] && b ∈ [2;9] where a*b <= 999

    Diff: a-b=c
        Is not possible because the difference between a 2-digit number and a 1-digit number
        is at most 90 but c is greater than 99.

    Division: a / b = c
        Same as the difference

    :return: An array with all three digit result solutions
    """

    result = []
    # Summation first
    for a in range(91, 100):
        for b in range(100 - a, 10):
            c = a + b
            result.append("{}+{}={}".format(a, b, c))
            result.append("{}+{}={}".format(b, a, c))  # Commutation

    # Multiplication first
    for a in range(12, 99):
        for b in range(ceil(100 / a), 10):
            c = a * b
            if c > 999: break
            result.append("{}*{}={}".format(a, b, c))
            result.append("{}*{}={}".format(b, a, c))  # Commutation

    return result


def generate_2d_result_solutions():
    """
        Since nerdle has 8 digit solutions for results to be 2 digit numbers we have
        some different open formulas which is basically a tree generated by the combinatorics
        of the operations plus the possible commutations.

        With 1 operation
            Sum         aa+bb | bb+aa
            Mul         Not possible
            Dif         aa-bb
            Div         Not possible

        With 2 operations
            Sum - Sum
            Sum - Dif | Dif - Sum
            Sum - Div | Div - Sum
            Sum - Mul | Mul - Sum

            Dif - Dif                 Not possible
            Dif - Div | Div - Dif     Not possible
            Dif - Mul | Mul - Dif

            Div - Div                 Not possible
            Div - Mul | Mul - Div

            Mul - Mul

        :return: An array with all three digit result solutions
    """

    result = []

    # With one operator
    # aa+bb = bb+aa = cc
    for a in range(10, 100):
        for b in range(10, 100 - a):
            c = a + b
            result.append("{}+{}={}".format(a, b, c))
            result.append("{}+{}={}".format(b, a, c))  # Commutation

    # aa*bb = bb*aa = cc is not a possible operation since the least number with two digits is 10 and 10*10 > 99.
    # then it follows that for every a and b, a*b yields a number greater than 99

    # aa-bb = cc
    for a in range(10, 90):
        for b in range(10 + a, 100):
            c = b - a
            result.append("{}-{}={}".format(b, a, c))

    # aa/bb = cc is not a possible subset of solutions since the greatest cc obtainable is max(a)/min(b)
    # which yields 99/10 = 9.9 but c > 10.

    # With two operators

    # Sum - Sum
    # a+b+c = a+c+b = b+a+c = b+c+a = c+a+b = c+b+a = dd
    for a in range(1, 10):
        for b in range(1, 10):
            for c in range(10 - (a + b), 10):
                d = a + b + c
                result.append("{}+{}+{}={}".format(a, b, c, d))

    # Sum - Difference | Difference - Sum
    # a+b-c = b+a-c = a-c+b = b-c+a = dd
    for c in range(0, 9):
        for a in range(c, 10):
            for b in range(10 + c - a, 10):
                if a < 0:
                    continue
                d = a + b - c
                result.append("{}+{}-{}={}".format(a, b, c, d))
                result.append("{}-{}+{}={}".format(a, c, b, d))

    # Sum - Div | Div - Sum
    # a+b/c = b/c+a = dd
    for a in range(1, 10):
        for b in range(10 - a, 10):
            for c in range(1, b + 1):
                if a + b / c < 10:
                    break

                if b % c != 0:
                    continue

                d = a + (b / c)
                result.append("{}+{}/{}={}".format(a, b, c, int(d)))
                result.append("{}/{}+{}={}".format(b, c, a, int(d)))

    # Sum - Mul | Mul - Sum
    # a+b*c = b*c+a = dd
    for a in range(0, 10):
        for b in range(0, 10):
            for c in range(0, 10):
                d = a+b*c
                if d > 99:
                    break
                if d < 10:
                    continue

                result.append("{}+{}*{}={}".format(a, b, c, int(d)))
                result.append("{}*{}+{}={}".format(b, c, a, int(d)))

    # Dif - Dif
    # Since the greater value obtainable from the difference of three numbers of one digit is 9-0-0=9
    # Then it follows that the set generated with two dif operations is not in the solution set

    # Dif - Div | Div - Dif
    # Same argument as dif dif

    # Dif - Mul | Mul - Dif
    # While Dif - Mul is not possible, Mul - Dif is.
    # a-b*c = b*c-a = dd
    for b in range(0, 10):
        for c in range(0, 10):
            for a in range(0, 10):
                d = b*c-a
                if d < 0:
                    break

                if d < 10:
                    continue

                result.append("{}*{}-{}={}".format(b, c, a, d))

    # Div - Div
    # Same as Dif - Dif


    # Div - Mul | Mul - Div
    # Div - Mul is not possible, but Mul - Div is
    # b*c/a = dd
    for b in range(1, 10):
        for c in range(1, 10):
            if b*c < 10:
                continue

            for a in range(1, 10):
                if b*c % a != 0:
                    continue

                d = b*c / a
                if d < 10:
                    break

                result.append("{}*{}/{}={}".format(b, c, a, int(d)))


    # Mul - Mul
    for a in range(1, 10):
        for b in range(1, 10):
            for c in range(1, 10):
                d = a * b * c
                if d < 10:
                    continue

                if d > 100:
                    break

                result.append("{}*{}*{}={}".format(a, b, c, d))

    return result


def generate_1d_result_solutions():
    """
    Left without implementation. It is similar to 2d solutions, but larger.
    For the purpose of this program I will work only with the solution set with 3 and 2 digit results.

    :return:
    """


if __name__ == "__main__":
    solutions = generate_3d_result_solutions()
    solutions += generate_2d_result_solutions()

    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    f = open(POSSIBLE_WORDS_LIST_FILE, "w")
    for s in solutions:
        f.write(s + "\n")
    f.close()
