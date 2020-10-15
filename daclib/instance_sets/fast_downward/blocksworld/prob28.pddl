

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b7)
(on b2 b11)
(on b3 b1)
(on b4 b12)
(on b5 b8)
(on b6 b3)
(on b7 b10)
(on-table b8)
(on b9 b6)
(on-table b10)
(on b11 b9)
(on b12 b5)
(clear b2)
(clear b4)
)
(:goal
(and
(on b2 b10)
(on b3 b7)
(on b4 b11)
(on b5 b4)
(on b6 b8)
(on b7 b2)
(on b9 b1)
(on b11 b9))
)
)


