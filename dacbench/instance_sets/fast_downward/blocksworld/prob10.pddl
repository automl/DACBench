

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(arm-empty)
(on b1 b7)
(on-table b2)
(on-table b3)
(on b4 b1)
(on-table b5)
(on b6 b8)
(on b7 b11)
(on b8 b3)
(on-table b9)
(on b10 b9)
(on-table b11)
(clear b2)
(clear b4)
(clear b5)
(clear b6)
(clear b10)
)
(:goal
(and
(on b1 b3)
(on b3 b8)
(on b4 b11)
(on b6 b4)
(on b7 b1)
(on b9 b6)
(on b10 b9)
(on b11 b2))
)
)


