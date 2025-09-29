

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b e)
(on c f)
(on d c)
(ontable e)
(on f b)
(clear a)
)
(:goal
(and
(on c e)
(on e a))
)
)


