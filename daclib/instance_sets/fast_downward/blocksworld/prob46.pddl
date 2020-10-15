

(define (problem BW-rand-14)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 )
(:init
(arm-empty)
(on-table b1)
(on-table b2)
(on b3 b13)
(on b4 b14)
(on b5 b2)
(on b6 b3)
(on b7 b9)
(on b8 b1)
(on b9 b10)
(on b10 b12)
(on b11 b5)
(on b12 b8)
(on b13 b11)
(on b14 b7)
(clear b4)
(clear b6)
)
(:goal
(and
(on b4 b6)
(on b5 b12)
(on b6 b2)
(on b7 b9)
(on b8 b5)
(on b11 b13)
(on b12 b11)
(on b14 b4))
)
)


