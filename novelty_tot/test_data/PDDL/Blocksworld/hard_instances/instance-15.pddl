

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(ontable b)
(on c b)
(on d c)
(on e a)
(ontable f)
(clear e)
(clear f)
)
(:goal
(and
(on b d)
(on d e))
)
)


