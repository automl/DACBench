

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(arm-empty)
(on-table b1)
(on b2 b7)
(on b3 b1)
(on b4 b9)
(on b5 b10)
(on b6 b3)
(on b7 b4)
(on-table b8)
(on b9 b11)
(on b10 b8)
(on b11 b5)
(clear b2)
(clear b6)
)
(:goal
(and
(on b1 b11)
(on b2 b7)
(on b5 b4)
(on b6 b2)
(on b7 b8)
(on b8 b3)
(on b9 b6)
(on b10 b1)
(on b11 b5))
)
)


