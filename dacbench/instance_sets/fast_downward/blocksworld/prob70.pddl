

(define (problem BW-rand-17)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 )
(:init
(arm-empty)
(on b1 b13)
(on-table b2)
(on b3 b1)
(on b4 b15)
(on b5 b11)
(on b6 b12)
(on-table b7)
(on b8 b6)
(on-table b9)
(on b10 b17)
(on b11 b4)
(on-table b12)
(on b13 b14)
(on b14 b16)
(on b15 b10)
(on b16 b5)
(on b17 b8)
(clear b2)
(clear b3)
(clear b7)
(clear b9)
)
(:goal
(and
(on b1 b6)
(on b3 b2)
(on b4 b7)
(on b5 b12)
(on b6 b8)
(on b8 b13)
(on b9 b10)
(on b10 b1)
(on b11 b17)
(on b12 b14)
(on b13 b4)
(on b16 b3))
)
)


