"""
Python 切片专项练习 · 9 道 LeetCode 风格题目

每道题都是一个 pytest test。VS Code 的 ▶ 按钮会出现在 def test_qN() 旁边。

知识点覆盖:
  1. 基本切片  lst[start:end]
  2. 步长切片  lst[start:end:step]
  3. 负索引    lst[-n:]
  4. 反转      lst[::-1]
  5. 字符串切片
  6. 负步长
  7. 省略边界
  8. 切片赋值（原地修改）
  9. 综合应用：循环位移
"""

import pytest


def test_q1():
    # 第1题：基本切片 — 取子列表
    def slice_range(lst: list, start: int, end: int) -> list:
        """
        返回 lst 中从索引 start 到 end（不含 end）的新列表。
        示例: slice_range([0,1,2,3,4,5], 2, 5) → [2, 3, 4]
        """
        # YOUR CODE HERE
        return lst[start:end]
        ...

    data = [0, 1, 2, 3, 4, 5, 6, 7]
    out = slice_range(data, 2, 5)
    assert out == _expected_q1()
    # 确保没改原 list
    assert data == [0, 1, 2, 3, 4, 5, 6, 7], "不要修改原 list！"


def test_q2():
    # 第2题：步长切片 — 每隔一个取一个
    def every_other(lst: list) -> list:
        """
        返回新列表，只包含原列表索引为偶数的元素（即 lst[0], lst[2], lst[4]...）。
        示例: every_other([10, 20, 30, 40, 50]) → [10, 30, 50]
        """
        # YOUR CODE HERE
        return lst[::2]
        ...

    out = every_other([10, 20, 30, 40, 50])
    assert out == _expected_q2()


def test_q3():
    # 第3题：负索引 — 取最后 N 个元素
    def last_n(lst: list, n: int) -> list:
        """
        返回 lst 的最后 n 个元素。
        提示: 当 n=0 时应返回空列表。
        示例: last_n([1, 2, 3, 4, 5], 3) → [3, 4, 5]
        """
        # YOUR CODE HERE
        return lst[-n:]

    data = [1, 2, 3, 4, 5, 6, 7, 8]
    assert last_n(data, 3) == _expected_q3a()
    assert last_n(data, 0) == _expected_q3b()
    assert last_n(data, 8) == _expected_q3c()


def test_q4():
    # 第4题：反转 — 用切片反转列表
    def reverse_list(lst: list) -> list:
        """
        返回反转后的新列表。不允许用 list.reverse() 或 reversed()。
        示例: reverse_list([1, 2, 3]) → [3, 2, 1]
        """
        # this is what starting to feel magical, and unintuitive.
        return lst[::-1]

    out = reverse_list([1, 2, 3, 4])
    assert out == _expected_q4()
    # 确认原 list 没变
    data = [1, 2, 3, 4]
    reverse_list(data)
    assert data == [1, 2, 3, 4], "不要修改原 list！"


def test_q5():
    # 第5题：字符串切片 — 回文判断
    def is_palindrome(s: str) -> bool:
        """
        用切片判断字符串是否回文（正着读反着读一样）。
        示例: is_palindrome("racecar") → True
              is_palindrome("hello")   → False
        """
        # YOUR CODE HERE
        s = s.replace(" ", "").lower()
        return s == s[::-1]

    assert is_palindrome("racecar") == _expected_q5a()
    assert is_palindrome("hello") == _expected_q5b()
    assert is_palindrome("a") == _expected_q5c()
    assert is_palindrome("") == _expected_q5d()


def test_q6():
    # 第6题：负步长 — 反向跳步取
    def reverse_step(lst: list, step: int) -> list:
        """
        从后往前，每隔 step 个元素取一个。
        示例: reverse_step([1, 2, 3, 4, 5, 6, 7, 8], 2) → [8, 6, 4, 2]
              （从 8 开始，往前跳 2 步取 6，再跳 2 步取 4...）
        """
        # YOUR CODE HERE
        return lst[::-step]

    assert reverse_step([1, 2, 3, 4, 5, 6, 7, 8], 2) == _expected_q6a()
    assert reverse_step([1, 2, 3, 4, 5], 3) == _expected_q6b()


def test_q7():
    # 第7题：省略边界 — 去掉首尾各 N 个
    def drop_first_last(lst: list, n: int) -> list:
        """
        去掉前 n 个和后 n 个元素，返回中间部分。
        如果 n*2 >= len(lst)，返回空列表。
        示例: drop_first_last([0,1,2,3,4,5,6,7], 3) → [3, 4]
        """
        # YOUR CODE HERE
        if n * 2 >= len(lst):
            return []
        return lst[n:-n]

    data = [0, 1, 2, 3, 4, 5, 6, 7]
    assert drop_first_last(data, 3) == _expected_q7a()
    assert drop_first_last(data, 4) == _expected_q7b()
    assert drop_first_last(data, 0) == _expected_q7c()


def test_q8():
    # 第8题：切片赋值 — 原地替换中间部分
    def replace_middle_in_place(lst: list, replacement: list) -> None:
        """
        原地修改 lst: 将 lst 中间 1/3 部分替换为 replacement。
        假设 len(lst) 是 3 的倍数。「中间 1/3」指索引 n/3 到 2n/3 的区域。
        示例:
          data = [0,1,2,3,4,5,6,7,8]    # len=9, n/3=3, 2n/3=6
          replace_middle_in_place(data, [99, 99, 99])
          → data 变为 [0,1,2,99,99,99,6,7,8]
        """
        # YOUR CODE HERE
        n = len(lst)
        start = n // 3
        end = 2 * n // 3
        lst[start:end] = replacement

    data1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    replace_middle_in_place(data1, [99, 99, 99])
    assert data1 == _expected_q8a()

    data2 = [10, 20, 30, 40, 50, 60]
    replace_middle_in_place(data2, [99, 99])
    assert data2 == _expected_q8b()


def test_q9():
    # 第9题：综合 — 循环右移 k 位
    def rotate(lst: list, k: int) -> list:
        """
        返回新列表，将 lst 循环右移 k 位。
        只能用切片和列表拼接，不允许用循环。
        k 可以是任意非负整数（提示: 先对 len(lst) 取模）。
        示例: rotate([1, 2, 3, 4, 5], 2) → [4, 5, 1, 2, 3]
              rotate([1, 2, 3], 5)       → [2, 3, 1]  (因为 5 % 3 = 2)
        """
        # YOUR CODE HERE
        k = k % len(lst) if lst else 0
        return lst[-k:] + lst[:-k]

    assert rotate([1, 2, 3, 4, 5], 2) == _expected_q9a()
    assert rotate([1, 2, 3], 5) == _expected_q9b()
    assert rotate([1, 2, 3, 4], 0) == _expected_q9c()
    assert rotate([1], 100) == _expected_q9d()


# ---------------------------------------------------------------------------
#  Expected values (sealed away so you don't see the answer while implementing).
# ---------------------------------------------------------------------------


def _expected_q1():
    return [2, 3, 4]


def _expected_q2():
    return [10, 30, 50]


def _expected_q3a():
    return [6, 7, 8]


def _expected_q3b():
    return []


def _expected_q3c():
    return [1, 2, 3, 4, 5, 6, 7, 8]


def _expected_q4():
    return [4, 3, 2, 1]


def _expected_q5a():
    return True


def _expected_q5b():
    return False


def _expected_q5c():
    return True


def _expected_q5d():
    return True


def _expected_q6a():
    return [8, 6, 4, 2]


def _expected_q6b():
    return [5, 2]


def _expected_q7a():
    return [3, 4]


def _expected_q7b():
    return []


def _expected_q7c():
    return [0, 1, 2, 3, 4, 5, 6, 7]


def _expected_q8a():
    return [0, 1, 2, 99, 99, 99, 6, 7, 8]


def _expected_q8b():
    return [10, 20, 99, 99, 50, 60]


def _expected_q9a():
    return [4, 5, 1, 2, 3]


def _expected_q9b():
    return [2, 3, 1]


def _expected_q9c():
    return [1, 2, 3, 4]


def _expected_q9d():
    return [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
