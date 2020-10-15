

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(arm-empty)
(on b1 b3)
(on b2 b7)
(on b3 b5)
(on b4 b10)
(on b5 b4)
(on-table b6)
(on b7 b6)
(on b8 b2)
(on-table b9)
(on-table b10)
(on b11 b1)
(clear b8)
(clear b9)
(clear b11)
)
(:goal
(and
(on b1 b8)
(on b4 b5)
(on b5 b1)
(on b7 b4)
(on b8 b6)
(on b9 b7)
(on b10 b9)
(on b11 b10))
)
)


