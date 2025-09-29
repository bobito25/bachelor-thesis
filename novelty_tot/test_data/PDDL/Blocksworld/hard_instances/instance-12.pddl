

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(ontable b)
(ontable c)
(ontable d)
(on e c)
(on f e)
(clear a)
(clear d)
(clear f)
)
(:goal
(and
(on a b)
(on c d)
(on f e))
)
)


