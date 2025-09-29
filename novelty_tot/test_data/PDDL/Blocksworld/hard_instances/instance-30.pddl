

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(ontable b)
(on c b)
(ontable d)
(on e f)
(ontable f)
(clear a)
(clear d)
(clear e)
)
(:goal
(and
(on a b)
(on b d)
(on c a)
(on d f)
(on f e))
)
)


