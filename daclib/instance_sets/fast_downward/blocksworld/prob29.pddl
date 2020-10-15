

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b11)
(on b2 b7)
(on b3 b5)
(on-table b4)
(on b5 b2)
(on b6 b3)
(on b7 b9)
(on-table b8)
(on b9 b1)
(on b10 b4)
(on b11 b10)
(on-table b12)
(clear b6)
(clear b8)
(clear b12)
)
(:goal
(and
(on b1 b8)
(on b3 b2)
(on b4 b12)
(on b5 b7)
(on b6 b11)
(on b7 b6)
(on b8 b3)
(on b10 b4)
(on b11 b9))
)
)


