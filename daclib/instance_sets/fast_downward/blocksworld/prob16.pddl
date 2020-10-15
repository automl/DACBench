

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(arm-empty)
(on b1 b10)
(on b2 b4)
(on b3 b6)
(on b4 b11)
(on-table b5)
(on b6 b5)
(on b7 b2)
(on b8 b7)
(on-table b9)
(on b10 b9)
(on b11 b1)
(clear b3)
(clear b8)
)
(:goal
(and
(on b1 b4)
(on b3 b2)
(on b4 b9)
(on b5 b11)
(on b6 b5)
(on b7 b10)
(on b9 b6)
(on b10 b1)
(on b11 b8))
)
)


