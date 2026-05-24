#! /usr/bin/env python3
# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    vector.py                                         :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: maprunty <maprunty@student.42heilbronn.d  +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2026/01/31 01:37:00 by maprunty         #+#    #+#              #
#    Updated: 2026/05/24 08:12:01 by maprunty        ###   ########.fr        #
#                                                                             #
# *************************************************************************** #
"""Module for vector and matrix classes and operations."""

from collections.abc import Iterator
from math import sqrt

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class Vec2:
    """Class for storing 2D Coords."""

    x: int | float = Field(default=0)
    y: int | float = Field(default=0)

    def normalized(self) -> "Vec2":
        """Return a normalized version of the vector."""
        mag = abs(self)
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return self / mag

    def __add__(self, other: "Vec2") -> "Vec2":
        """Add a vec2 instance with another."""
        return Vec2(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other: "Vec2") -> "Vec2":
        """Sub a vec2 instance with another."""
        return Vec2(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, scaler: int) -> "Vec2":
        """Multiply a vec2 instance by a scalar."""
        return Vec2(
            self.x * scaler,
            self.y * scaler,
        )

    def __truediv__(self, scalar: float) -> "Vec2":
        """Divide a vec2 instance by a scalar."""
        return Vec2(self.x / scalar, self.y / scalar)

    def __eq__(self, other: object) -> bool:
        """Equate a vec2 instance with another."""
        if not isinstance(other, Vec2):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __gt__(self, other: "Vec2") -> bool:
        """Check if a vec2 instance is greater than another."""
        return abs(self) >= abs(other)

    def __ge__(self, other: "Vec2") -> bool:
        """Check if a vec2 instance is greater than or equal to another."""
        return self > other or self == other

    def __lt__(self, other: "Vec2") -> bool:
        """Check if a vec2 instance is less than another."""
        return abs(self) <= abs(other)

    def __le__(self, other: "Vec2") -> bool:
        """Check if a vec2 instance is less than or equal to another."""
        return self < other or self == other

    def __abs__(self) -> float:
        """Return magnitude of a vector."""
        return sqrt(self.x**2 + self.y**2)

    def __repr__(self) -> str:
        """Return a tuple represantation of a Vec2 instance."""
        cls = self.__class__.__name__
        return f"{cls}(x={self.x}, y={self.y})"

    def __str__(self) -> str:
        """Return a str tuple represantation of a Vec2 instance."""
        return f"{self.x},{self.y}"

    def __iter__(self) -> Iterator[float | int]:
        """Iterate over the fields of a Vec2 instance."""
        return iter((self.x, self.y))

    def __hash__(self) -> int:
        """Return a hash of a Vec2 instance."""
        return hash((self.x, self.y))


@dataclass
class Vecn:
    """Class for storing n-dimensional Coords."""

    coords: list[int | float] = Field(default_factory=list)

    def normalized(self) -> "Vecn":
        """Return a normalized version of the vector."""
        mag = abs(self)
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return self / mag

    def __add__(self, other: "Vecn") -> "Vecn":
        """Add a vecn instance with another."""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same number of dimensions")
        return Vecn(
            [a + b for a, b in zip(self.coords, other.coords, strict=True)]
        )

    def __sub__(self, other: "Vecn") -> "Vecn":
        """Sub a vecn instance with another."""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same number of dimensions")
        return Vecn(
            [a - b for a, b in zip(self.coords, other.coords, strict=True)]
        )

    def __mul__(self, scaler: int) -> "Vecn":
        """Multiply a vecn instance by a scalar."""
        return Vecn([coord * scaler for coord in self.coords])

    def __rmul__(self, scaler: int) -> "Vecn":
        """Multiply a vecn instance by a scalar."""
        return self * scaler

    def dot(self, other: "Vecn") -> int | float:
        """Return the dot product of this vector with another."""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same number of dimensions")
        return sum(
            a * b for a, b in zip(self.coords, other.coords, strict=True)
        )

    def __matmul__(self, other: "Vecn") -> int | float:
        """Return the dot product of this vector with another."""
        return self.dot(other)

    def cross(self, other: "Vecn") -> "Vecn":
        """Return the cross product of this vector with another."""
        if len(self) != 3 or len(other) != 3:
            raise ValueError(
                "Cross product is only defined for 3-dimensional vectors"
            )
        a1, a2, a3 = self.coords
        b1, b2, b3 = other.coords
        return Vecn(
            [
                a2 * b3 - a3 * b2,
                a3 * b1 - a1 * b3,
                a1 * b2 - a2 * b1,
            ]
        )

    def __truediv__(self, scalar: float) -> "Vecn":
        """Divide a vecn instance by a scalar."""
        return Vecn([coord / scalar for coord in self.coords])

    def __eq__(self, other: object) -> bool:
        """Equate a vecn instance with another."""
        if not isinstance(other, Vecn):
            return NotImplemented
        return self.coords == other.coords

    def __gt__(self, other: "Vecn") -> bool:
        """Check if a vecn instance is greater than another."""
        return abs(self) >= abs(other)

    def __ge__(self, other: "Vecn") -> bool:
        """Equate a vecn instance with another."""
        return self > other or self == other

    def __lt__(self, other: "Vecn") -> bool:
        """Check if a vecn instance is less than another."""
        return abs(self) <= abs(other)

    def __le__(self, other: "Vecn") -> bool:
        """Check if a vecn instance is less than or equal to another."""
        return self < other or self == other

    def __abs__(self) -> float:
        """Return magnitude of a vector."""
        return sqrt(sum(coord**2 for coord in self.coords))

    def __len__(self) -> int:
        """Return the number of dimensions of a Vecn instance."""
        return len(self.coords)

    def __iter__(self) -> Iterator[int | float]:
        """Iterate over the fields of a Vecn instance."""
        return iter(self.coords)

    def __repr__(self) -> str:
        """Return a tuple represantation of a Vecn instance."""
        cls = self.__class__.__name__
        return f"{cls}(coords={self.coords})"

    def __getitem__(self, index: int) -> int | float:
        """Return the coordinate at the given index."""
        return self.coords[index]

    def __setitem__(self, index: int, value: int | float) -> None:
        """Set the coordinate at the given index."""
        self.coords[index] = value

    def __str__(self) -> str:
        """Return a str tuple represantation of a Vecn instance."""
        return ",".join(str(coord) for coord in self.coords)

    def __hash__(self) -> int:
        """Return a hash of a Vecn instance."""
        return hash(tuple(self.coords))


@dataclass
class Matrix:
    """Class for storing a matrix."""

    rows: list[Vecn] = Field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of rows in a Matrix instance."""
        return len(self.rows)

    def __transpose__(self) -> "Matrix":
        """Return the transpose of a Matrix instance."""
        if not self.rows:
            return Matrix()
        num_cols = len(self)
        return Matrix(
            [
                Vecn([row.coords[i] for row in self.rows])
                for i in range(num_cols)
            ]
        )

    def transpose(self) -> "Matrix":
        """Return the transpose of a Matrix instance."""
        return self.__transpose__()

    def cols(self) -> list[Vecn]:
        """Return the columns of a Matrix instance."""
        return self.transpose().rows

    def swap_rows(self, i: int, j: int) -> None:
        """Swap two rows of a Matrix instance."""
        if i >= len(self) or j >= len(self):
            raise IndexError("Row index out of range")
        tmp = self[i]
        self[i] = self[j]
        self[j] = tmp

    def row_echelon(self) -> "Matrix":
        """Return a row echelon version of a Matrix instance.

        Step 1. Begin with an m × n matrix A. If A = 0, go to Step 7.
        Step 2. Determine the leftmost non-zero column.
        Step 3. Use elementary row operations to put a 1 in the topmost position
        (we call this position pivot position) of this column.
        Step 4. Use elementary row operations to put zeros (strictly) below the pivot
        position.
        Step 5. If there are no more non-zero rows (strictly) below the pivot position,
        then go to Step 7.
        Step 6. Apply Step 2-5 to the submatrix consisting of the rows that lie
        (strictly below) the pivot position.
        Step 7. The resulting matrix is in row-echelon form
        """
        if not self.rows:
            return Matrix()
        m = len(self)
        n = len(self.rows[0])
        pivot_row = 0
        for col in range(min(n, m)):
            r = pivot_row
            while r < m and self[r][col] == 0:
                r += 1
            if r == m:
                continue
            self.swap_rows(r, pivot_row)

            pivot = self[pivot_row][col]
            for c in range(col, n):
                self[pivot_row][c] /= pivot
            for r in range(pivot_row + 1, m):
                factor = self[r][col]
                for c in range(col, n):
                    self[r][c] -= factor * self[pivot_row][c]
            pivot_row += 1
        return self

    def __add__(self, other: "Matrix") -> "Matrix":
        """Add a matrix instance with another."""
        if len(self) != len(other.rows):
            raise ValueError("Matrices must have the same number of rows")
        return Matrix(
            [
                row1 + row2
                for row1, row2 in zip(self.rows, other.rows, strict=True)
            ]
        )

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Sub a matrix instance with another."""
        if len(self.rows) != len(other.rows):
            raise ValueError("Matrices must have the same number of rows")
        return Matrix(
            [
                row1 - row2
                for row1, row2 in zip(self.rows, other.rows, strict=True)
            ]
        )

    def __mul__(self, scalar: int) -> "Matrix":
        """Multiply a matrix instance by a scalar."""
        return Matrix([row * scalar for row in self.rows])

    def __rmul__(self, scalar: int) -> "Matrix":
        """Multiply a matrix instance by a scalar."""
        return self * scalar

    def dot(self, other: "Matrix") -> "Matrix":
        """Return the dot product of this matrix with another."""
        if len(self.cols()) != len(other.rows):
            raise ValueError(
                "Number of columns in the first matrix must equal the number of rows in the second matrix"
            )
        return Matrix(
            [
                Vecn(
                    [
                        sum(
                            a * b
                            for a, b in zip(
                                row.coords, col.coords, strict=True
                            )
                        )
                        for col in other.cols()
                    ]
                )
                for row in self.rows
            ]
        )

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Return the dot product of this matrix with another."""
        return self.dot(other)

    def __eq__(self, other: object) -> bool:
        """Equate a matrix instance with another."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.rows == other.rows

    def __gt__(self, other: "Matrix") -> bool:
        """Check if a matrix instance is greater than another."""
        return abs(self) >= abs(other)

    def __ge__(self, other: "Matrix") -> bool:
        """Check if a matrix instance is greater than or equal to another."""
        return self > other or self == other

    def __lt__(self, other: "Matrix") -> bool:
        """Check if a matrix instance is less than another."""
        return abs(self) <= abs(other)

    def __le__(self, other: "Matrix") -> bool:
        """Check if a matrix instance is less than or equal to another."""
        return self < other or self == other

    def __abs__(self) -> float:
        """Return magnitude of a matrix."""
        return sqrt(sum(abs(row) ** 2 for row in self.rows))

    def __iter__(self) -> Iterator[Vecn]:
        """Iterate over the rows of a Matrix instance."""
        return iter(self.rows)

    def __getitem__(self, index: int) -> Vecn:
        """Return the row at the given index."""
        return self.rows[index]

    def __setitem__(self, index: int, value: Vecn) -> None:
        """Set the row at the given index."""
        self.rows[index] = value

    def __repr__(self) -> str:
        """Return a tuple represantation of a Matrix instance."""
        cls = self.__class__.__name__
        return f"{cls}(rows={self.rows})"

    def __str__(self) -> str:
        """Return a str tuple represantation of a Matrix instance."""
        return "\n".join("|" + str(row) + "|" for row in self.rows)

    def __hash__(self) -> int:
        """Return a hash of a Matrix instance."""
        return hash(tuple(self.rows))


if __name__ == "__main__":
    # Example usage
    v1 = Vec2(3, 4)
    v2 = Vec2(1, 2)
    print(v1 + v2)  # Vec2(x=4, y=6)
    print(v1 - v2)  # Vec2(x=2, y=2)
    print(v1 * 2)  # Vec2(x=6, y=8)
    print(v1 / 5)  # Vec2(x=0.6, y=0.8)
    print(abs(v1))  # 5.0
    print(v1.normalized())  # Vec2(x=0.6, y=0.8)

    m1 = Matrix(rows=[Vecn([1, 2]), Vecn([3, 4])])
    m2 = Matrix(rows=[Vecn([5, 6]), Vecn([7, 8])])
    m3 = Matrix(rows=[Vecn([2, 1, -1]), Vecn([-3, -1, 2]), Vecn([-2, 1, 2])])
    print(m1 + m2)  # Matrix(rows=[Vecn(coords=[6, 8]), Vecn(coords=[10, 12])])
    print(
        m1 - m2
    )  # Matrix(rows=[Vecn(coords=[-4, -4]), Vecn(coords=[-4, -4])])
    print(m1 * 3)  # Matrix(rows=[Vecn(coords=[3, 6]), Vecn(coords=[9, 12])])
    print(
        m1.dot(m2)
    )  # Matrix(rows=[Vecn(coords=[19, 22]), Vecn(coords=[43, 50])])
    print(
        m1.transpose()
    )  # Matrix(rows=[Vecn(coords=[1, 3]), Vecn(coords=[2, 4])])
    print(
        m1.row_echelon()
    )  # Matrix(rows=[Vecn(coords=[1.0, 2.0]), Vecn(coords=[0.0, 1.0])])
    print(m3, "\n")
    print(m3.row_echelon(), "\n")
