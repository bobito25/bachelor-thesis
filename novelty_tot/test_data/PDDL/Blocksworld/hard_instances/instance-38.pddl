

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(ontable b)
(on c f)
(on d e)
(on e b)
(on f d)
(clear a)
)
(:goal
(and
(on a e)
(on c b)
(on e c)
(on f d))
)
)


