

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(ontable b)
(on c b)
(on d c)
(on e f)
(ontable f)
(clear a)
(clear e)
)
(:goal
(and
(on b f)
(on c b)
(on d e))
)
)


